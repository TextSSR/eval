'''
Part of the implementation is borrowed and modified from TextDiffuser, publicly available at https://github.com/microsoft/unilm/blob/master/textdiffuser/inference.py
'''
import os
import json
import random
from tqdm import tqdm
import argparse
import numpy as np
from packaging import version
from termcolor import colored
from PIL import Image
from datasets import disable_caching
import torch
import torch.utils.checkpoint
from torchvision import transforms
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import ImageDraw
import cv2
import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

disable_caching()
check_min_version("0.15.0.dev0")
PLACE_HOLDER = '*'
data_type = "ic13"

ckpt_path = '../ckpt/textdiffuser2-full-ft-inpainting'
json_path = f'../benchmark/{data_type}/test.json'
output_dir = f'./out/{data_type}'
input_dir = f'../benchmark/{data_type}/images'

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='runwayml/stable-diffusion-v1-5', # no need to modify this  
        # default='stabilityai/stable-diffusion-2-1',  # no need to modify this
        help="Path to pretrained model or model identifier from huggingface.co/models. Please do not modify this.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text-to-image-with-template",
        choices=["text-to-image-with-template"],
        help="Three modes can be used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="output path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--classifier_free_scale",
        type=float,
        default=9.0,  # following stable diffusion (https://github.com/CompVis/stable-diffusion)
        help="Classifier free scale following https://arxiv.org/abs/2207.12598.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=ckpt_path,
        help='path of model'
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        default=True,
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=20,
        help="Diffusion steps for sampling."
    )
    parser.add_argument(
        "--vis_num",
        type=int,
        default=1,
        help="Number of images to be sample. Please decrease it when encountering out of memory error."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=77,
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
    print(f'{colored("[√]", "green")} Arguments are loaded.')
    return args


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

def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(os.path.join(args.output_dir, "full"))
        os.makedirs(os.path.join(args.output_dir, "region"))
    seed = args.seed if args.seed is not None else random.randint(0, 1000000)
    set_seed(seed)
    # Load scheduler, tokenizer and models.
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
        #### additional tokens are introduced, including coordinate tokens and character tokens
    print('***************')
    print(len(tokenizer))
    for i in range(520):
        tokenizer.add_tokens(['l' + str(i) ]) # left
        tokenizer.add_tokens(['t' + str(i) ]) # top
        tokenizer.add_tokens(['r' + str(i) ]) # width
        tokenizer.add_tokens(['b' + str(i) ]) # height    
    for c in alphabet:
        tokenizer.add_tokens([f'[{c}]']) 
    print(len(tokenizer))
    print('***************')

    if args.max_length == 77:
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path, subfolder="text_encoder", ignore_mismatched_sizes=True
        ).cuda()
    else:
        #### enlarge the context length of text encoder. empirically, enlarging the context length can proceed longer sequence. However, we observe that it will be hard to render general objects
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_path, subfolder="text_encoder", max_position_embeddings=args.max_length, ignore_mismatched_sizes=True
        ).cuda()

    text_encoder.resize_token_embeddings(len(tokenizer))

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").cuda()
    unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet"
    ).cuda()
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    # setup schedulers
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.set_timesteps(args.sample_steps)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    # inference loop
    data_list = load_data(args.json_path)
    words_dict = {}
    for j in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, j)
        img_name = item_dict['img_name']

        if os.path.exists(os.path.join(args.output_dir, img_name)):
            continue
        input_image_path = os.path.join(args.input_dir, img_name)

        ini_image = Image.open(input_image_path)


        original_size = ini_image.size  
        ini_image = ini_image.resize((512, 512))
        caption = item_dict['caption']
        caption_ids = tokenizer(
            caption, truncation=True, return_tensors="pt"
        ).input_ids[0].tolist()
        texts = item_dict['texts']
        for i, polygon in enumerate(item_dict['polygons']):
            i_name = str(i) + "_" + img_name
            text = texts[i]
            words_dict[i_name] = text
            ocr_ids = []
            polygon, x, y, w, h = get_bounding_rect(polygon, original_size)
            x_min, y_min, x_max, y_max = x, y, x+w, y+h
            char_list = list(text)
            ocr_ids.extend([f'l{x_min//4}', f't{y_min//4}', f'r{x_max//4}', f'b{y_max//4}'])
            char_list = [f'[{i}]' for i in char_list]
            ocr_ids.extend(char_list)
            ocr_ids.append(tokenizer.eos_token_id)
            ocr_ids.append(tokenizer.eos_token_id) 
            ocr_ids = tokenizer.encode(ocr_ids)

            prompt = caption_ids + ocr_ids
            prompt = prompt[ : args.max_length]
            prompt = torch.Tensor([prompt]).long().cuda()

            mask = Image.new("L", (512, 512), 0)
            draw = ImageDraw.Draw(mask)
            polygon_int = [(int(x), int(y)) for x, y in polygon]
            draw.polygon(polygon_int, fill=255) 
            mask = mask.convert("L")
            mask = np.array(mask).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
            mask = torch.where(mask < 0.5, torch.tensor(0.0), torch.tensor(1.0)).unsqueeze(0)

            masked_image = transforms.ToTensor()(ini_image).sub_(0.5).div_(0.5).unsqueeze(0) * (1-mask).unsqueeze(1) 

            masked_image = masked_image.cuda()

            masked_feature = vae.encode(masked_image).latent_dist.sample()
            masked_feature = masked_feature * vae.config.scaling_factor                 
            feature_mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(64, 64), mode='nearest')
            feature_mask = feature_mask.cuda()
            shape = (1, vae.config.latent_channels, 64, 64)
            latents = randn_tensor(shape, generator=torch.manual_seed(20), device=torch.device("cuda")) * scheduler.init_noise_sigma
            encoder_hidden_states = text_encoder(prompt)[0]

            intermediate_images = []

            for t in tqdm(scheduler.timesteps):
                with torch.no_grad():
                    latents_input = latents 
                    scaled_input = scheduler.scale_model_input(latents_input, t) 
                    noise_pred = unet(
                        sample=scaled_input, 
                        timestep=t, 
                        encoder_hidden_states=encoder_hidden_states, 
                        feature_mask=feature_mask, 
                        masked_feature=masked_feature
                    ).sample  # b, 4, 64, 64
                    
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                    intermediate_images.append(latents) 

            # decode and visualization
            input = 1 / vae.config.scaling_factor * latents
            sample_images = vae.decode(input.float(), return_dict=False)[0]  # (b, 3, 512, 512)

            # save pred_img
            pred_image_list = []
            for image in sample_images.float():
                image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                pred_image_list.append(image)

            pred_image_list[0].save(os.path.join(args.output_dir, "full", i_name))

            crop_real = ini_image.crop((x, y, x + w, y + h))
            crop = pred_image_list[0].crop((x, y, x + w, y + h))
            crop_path = os.path.join(args.output_dir, "region", i_name)
            crop.save(crop_path)

    json_path = os.path.join(args.output_dir, "labels.json")
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(words_dict, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
