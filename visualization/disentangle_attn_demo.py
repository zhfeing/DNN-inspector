import os
import logging
from typing import Dict, Any, List
import argparse

import h5py
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.cuda
import torch.backends.cudnn
import torch.utils.data as data
import torchvision.transforms.functional as TF

import cv_lib.utils as cv_utils
from cv_lib.config_parsing import get_cfg
from cv_lib.augmentation import UnNormalize

from driver.data import build_eval_dataset


def draw_img(img: torch.Tensor, save_fp: str):
    img = TF.to_pil_image(img)
    img.save(save_fp)


def disentangle_attn(raw_attn: torch.Tensor, start_id: int):
    raw_attn = raw_attn[start_id:, start_id:]
    attn_symmetric = (raw_attn + raw_attn.T) / 2
    attn_skew = (raw_attn - raw_attn.T) / 2

    ret = {
        "raw_attn": raw_attn,
        "symmetric": attn_symmetric,
        "skew": attn_skew
    }
    return ret


def analysis_mhsa(attentions: Dict[str, torch.Tensor], save_path: str, img_id: int, start_id: int):
    logger = logging.getLogger("analysis_mhsa")
    save_path = os.path.join(save_path, f"mhsa_{img_id}")
    os.makedirs(save_path, exist_ok=True)

    def vis_attn(attn: torch.Tensor, path: str):
        os.makedirs(path, exist_ok=True)
        attn_dict = disentangle_attn(attn, start_id)
        for k, attn in attn_dict.items():
            logger.info("%s attn: %s, F_norm: %.5f", path, k, attn.norm("fro").item())
            attn = attn.cpu().numpy()
            sns.heatmap(attn, cmap="coolwarm", xticklabels=False, yticklabels=False)
            fp = os.path.join(path, f"{k}.png")
            plt.savefig(fp, bbox_inches="tight")
            plt.close()

    for name, mhsa in attentions.items():
        mhsa = mhsa.squeeze()
        path = os.path.join(save_path, f"{name}")
        assert mhsa.dim() == 3
        mean_attn = mhsa.mean(dim=0)
        attn_head = mhsa.unbind(dim=0)
        vis_attn(mean_attn, os.path.join(path, "mean"))
        for h_id, attn in enumerate(attn_head):
            vis_attn(attn, os.path.join(path, f"head-{h_id}"))


def main(args):
    """
    What created in this function is only used in this process and not shareable
    """
    logger = cv_utils.get_root_logger(logging.INFO, os.path.join(args.save_path, "disentangle.log"))
    # split configs
    data_cfg: Dict[str, Any] = get_cfg(args.data_cfg)

    # set cuda
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        cv_utils.make_deterministic(args.seed)

    # get dataloader
    logger.info("Building dataset...")
    val_dataset, _, _ = build_eval_dataset(data_cfg)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    sampler = data.RandomSampler(val_dataset, generator=generator)
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=0,
        sampler=sampler
    )
    mean = val_dataset.dataset_mean
    std = val_dataset.dataset_std
    un_norm = UnNormalize(mean, std)

    # create model
    logger.info("Building model...")
    model: torch.jit.RecursiveScriptModule = torch.jit.load(args.jit, map_location="cpu")
    

    model.to(device)
    with torch.no_grad():
        model.to(device)
        i = 1
        for x, gt in tqdm.tqdm(val_loader, total=args.total):
            if i > args.total:
                break
            x = x.to(device)
            _, feats = model(x)
            mid_attn = {n: feats[n] for n in args.raw_attn_names}

            img_id = gt["image_id"].item()
            img = un_norm(x[0])
            draw_img(img, os.path.join(args.save_path, f"{img_id}.png"))
            analysis_mhsa(mid_attn, args.save_path, img_id, start_id=args.start_id)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--total", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--raw_attn_names", nargs="+", default=list())
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
