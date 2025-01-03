"""
Part of the implementation is borrowed and modified from GlyphControl, publicly available at https://github.com/AIGText/GlyphControl-release/blob/main/inference.py
"""
import torch
from PIL import Image
from cldm.hack import disable_verbosity, enable_sliced_attention
from scripts.rendertext_tool import load_model_from_config
from omegaconf import OmegaConf
import argparse
import os
import json
import random
import einops
from tqdm import tqdm
import numpy as np
import cv2
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from cldm.ddim_hacked import DDIMSampler
from torchvision.transforms import ToTensor

data_type = "ic13"

ckpt_path = '../ckpt/laion10M_epoch_6_model_ema_only.ckpt'
json_path = f'../benchmark/{data_type}/test.json'
output_dir = f'./out/{data_type}'
input_dir = f'../benchmark/{data_type}/images'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config.yaml",
        help="path to model config",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=ckpt_path,
        help='path to checkpoint of model'
    )
    parser.add_argument(
        "--save_memory",
        action="store_true",
        default=False,
        help="whether to save memory by transferring some unused parts of models to the cpu device during inference",
    )
    # specify the inference settings
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default='best quality, extremely detailed',
        help="additional prompt"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark',
        help="negative prompt"
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=512,
        help="image resolution",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="control strength",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="classifier-free guidance scale",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="ddim steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="seed",
    )
    parser.add_argument(
        "--guess_mode",
        action="store_true",
        help="whether use guess mode",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0,
        help="eta",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="output path"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=input_dir,
        help="path of glyph images from anytext evaluation dataset"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=json_path,
        help="json path for evaluation dataset"
    )
    args = parser.parse_args()
    return args


class Render_Text:
    def __init__(self,
                 model,
                 precision_scope=nullcontext,
                 transform=ToTensor(),
                 save_memory=False,
                 ):
        self.model = model
        self.precision_scope = precision_scope
        self.transform = transform
        self.ddim_sampler = DDIMSampler(model)
        self.save_memory = save_memory

    def process_multi(self,
                      shared_prompt,
                      glyph_img_path,
                      shared_num_samples, shared_image_resolution,
                      shared_ddim_steps, shared_guess_mode,
                      shared_strength, shared_scale, shared_seed,
                      shared_eta, shared_a_prompt, shared_n_prompt,
                      only_show_rendered_image=False,
                      font_name="calibri"
                      ):
        if shared_seed == -1:
            shared_seed = random.randint(0, 65535)
        seed_everything(shared_seed)
        with torch.no_grad(), self.precision_scope("cuda"), self.model.ema_scope("Sampling on Benchmark Prompts"):
            whiteboard_img = Image.open(glyph_img_path).convert("RGB")
            whiteboard_img = whiteboard_img.resize((shared_image_resolution, shared_image_resolution))
            control = self.transform(whiteboard_img.copy())
            if torch.cuda.is_available():
                control = control.cuda()
            control = torch.stack([control for _ in range(shared_num_samples)], dim=0)
            control = control.clone()
            control = [control]

            H, W = shared_image_resolution, shared_image_resolution
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", False)
                self.model.low_vram_shift(is_diffusing=False)

            print("control is None: {}".format(control is None))
            if shared_prompt.endswith("."):
                if shared_a_prompt == "":
                    c_prompt = shared_prompt
                else:
                    c_prompt = shared_prompt + " " + shared_a_prompt
            elif shared_prompt.endswith(","):
                if shared_a_prompt == "":
                    c_prompt = shared_prompt[:-1] + "."
                else:
                    c_prompt = shared_prompt + " " + shared_a_prompt
            else:
                if shared_a_prompt == "":
                    c_prompt = shared_prompt + "."
                else:
                    c_prompt = shared_prompt + ", " + shared_a_prompt
            cond_c_cross = self.model.get_learned_conditioning([c_prompt] * shared_num_samples)
            # print("prompt:", c_prompt)
            un_cond_cross = self.model.get_learned_conditioning([shared_n_prompt] * shared_num_samples)
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", True)
                self.model.low_vram_shift(is_diffusing=True)

            cond = {"c_concat": control, "c_crossattn": [cond_c_cross] if not isinstance(cond_c_cross, list) else cond_c_cross}
            un_cond = {"c_concat": None if shared_guess_mode else control, "c_crossattn": [un_cond_cross] if not isinstance(un_cond_cross, list) else un_cond_cross}
            shape = (4, H // 8, W // 8)

            if not self.model.learnable_conscale:
                self.model.control_scales = [shared_strength * (0.825 ** float(12 - i)) for i in range(13)] if shared_guess_mode else ([shared_strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            else:
                print("learned control scale: {}".format(str(self.model.control_scales)))
            samples, intermediates = self.ddim_sampler.sample(shared_ddim_steps, shared_num_samples,
                                                              shape, cond, verbose=False, eta=shared_eta,
                                                              unconditional_guidance_scale=shared_scale,
                                                              unconditional_conditioning=un_cond)
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", False)
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(shared_num_samples)]
        return results


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def load_data(input_path):
    content = load_json(input_path)
    d = []
    count = 0
    for gt in content['data_list']:
        info = {}
        info['img_name'] = gt['img_name']
        info['caption'] = gt['caption']

        if PLACE_HOLDER in info['caption']:
            count += 1
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if annotation['valid'] is False:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                pos.append(annotation['pos'])
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    if count > 0:
        print(f"Found {count} image's caption contain placeholder: {PLACE_HOLDER}, change to ' '...")
    return d

def get_item(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']
    item_dict['caption'] = cur_item['caption']
    item_dict['polygons'] = cur_item['polygons']
    item_dict['texts'] = cur_item['texts']
    return item_dict

def get_bounding_rect(polygon, original_size, new_size=(512, 512)):
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    polygon = np.array(polygon, dtype=np.float32)
    polygon[:, 0] = polygon[:, 0] * scale_x
    polygon[:, 1] = polygon[:, 1] * scale_y
    resized_polygon = np.array(polygon, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(resized_polygon)
    return resized_polygon, x, y, w, h

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(os.path.join(args.output_dir, "full"))
        os.makedirs(os.path.join(args.output_dir, "region"))
    disable_verbosity()
    if args.save_memory:
        enable_sliced_attention()
    cfg = OmegaConf.load(args.cfg)
    model = load_model_from_config(cfg, args.model_path, verbose=True)
    render_tool = Render_Text(model, save_memory=args.save_memory)
    if os.path.exists(args.output_dir) is not True:
        os.makedirs(args.output_dir)
    PLACE_HOLDER = '*'
    data_list = load_data(args.json_path)
    new_size = (args.image_resolution, args.image_resolution)
    words_dict = {}
    for i in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name']
        ini_img = cv2.imread(os.path.join(args.input_dir, img_name))
        height, width, _ = ini_img.shape
        original_size = (width, height)
        if os.path.exists(os.path.join(args.output_dir, img_name)):
            continue
        input_image_path = os.path.join(args.input_dir, img_name)
        results = render_tool.process_multi(item_dict["caption"], input_image_path, args.num_samples, args.image_resolution, args.ddim_steps, args.guess_mode, args.strength, args.scale, args.seed, args.eta, args.a_prompt, args.n_prompt)
        result = cv2.cvtColor(results[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(args.output_dir, "full", img_name), result)
        texts = item_dict['texts']
        for i, polygon in enumerate(item_dict['polygons']):
            i_name = str(i) + "_" + img_name
            words_dict[i_name] = texts[i]
            # Calculate bounding box
            polygon, x, y, w, h = get_bounding_rect(polygon, original_size, new_size)
            crop = results[0][y:y+h, x:x+w]
            # Save cropped images
            crop_path = os.path.join(args.output_dir, "region", i_name)
            os.makedirs(os.path.dirname(crop_path), exist_ok=True)
            cv2.imwrite(crop_path, crop)

    json_path = os.path.join(args.output_dir, "labels.json")
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(words_dict, json_file, ensure_ascii=False, indent=4)