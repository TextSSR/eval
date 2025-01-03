import torch
import random
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
from os.path import join as ospj
from torchvision.utils import save_image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from dataset.dataloader import get_dataloader

from util import *
from metrics import calc_fid, calc_lpips
import json

def predict(cfgs, model, sampler, batch):

    context = nullcontext if cfgs.aae_enabled else torch.no_grad
    
    with context():
        
        batch, batch_uc_1 = prepare_batch(cfgs, batch)

        c, uc_1 = model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc_1,
            force_uc_zero_embeddings=cfgs.force_uc_zero_embeddings,
        )
        
        x = sampler.get_init_noise(cfgs, model, cond=c, batch=batch, uc=uc_1)
        samples_z = sampler(model, x, cond=c, batch=batch, uc=uc_1, init_step=0,
                            aae_enabled = cfgs.aae_enabled, detailed = cfgs.detailed)

        samples_x = model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        return samples, samples_z


def test(model, sampler, dataloader, cfgs):
    
    output_dir = cfgs.output_dir
    os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    full_dir = ospj(output_dir, "full")
    crop_dir = ospj(output_dir, "region")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)

    results_json = {}
    cnt = 0
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        results, results_z = predict(cfgs, model, sampler, batch)
        result = results.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255  # detach 防止梯度问题

        for i, bbox in enumerate(batch["r_bbox"]):
            r_top, r_bottom, r_left, r_right = bbox
            i_name = str(i) + "_" + batch["image_name"][i]
            full = result[i]
            crop = full[r_top:r_bottom, r_left:r_right]

            Image.fromarray(crop.astype(np.uint8)).save(ospj(crop_dir, i_name))
            Image.fromarray(full.astype(np.uint8)).save(ospj(full_dir, i_name))
            
            results_json[i_name] = batch["label"][i]
            cnt += 1
    
    json_path = os.path.join(output_dir, "label.json")
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(results_json, json_file, ensure_ascii=False, indent=4)


    
    if cfgs.quan_test:
        calc_fid(fake_dir, real_dir)
        calc_lpips(fake_dir, real_dir)


if __name__ == "__main__":

    cfgs = OmegaConf.load("./configs/test.yaml")

    seed = random.randint(0, 2147483647)
    seed_everything(seed)

    model = init_model(cfgs)
    sampler = init_sampling(cfgs)
    dataloader = get_dataloader(cfgs, "val")

    test(model, sampler, dataloader, cfgs)

