import os
from typing import Dict, Any, List
import argparse

import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.cuda
import torch.backends.cudnn
import torch.utils.data as data
import captum.attr

import cv_lib.utils as cv_utils
from cv_lib.augmentation import UnNormalize

from driver.data import build_eval_dataset
from visualization.vis_feat_attn import vis_img


def vis_attr(
    attribution: torch.Tensor,
    save_path: str,
    img_id: int,
    feat_shape: List[int] = [14, 14],
    start_id: int = 1
):
    attribution = attribution.squeeze()
    assert attribution.dim() == 3

    attribution = attribution.detach().clamp_min(0).mean(dim=0)
    attribution = attribution.cpu().numpy()

    sns.heatmap(attribution, cmap="vlag", xticklabels=False, yticklabels=False)
    attr_fp = os.path.join(save_path, f"img_{img_id}_attr.png")
    plt.savefig(attr_fp, bbox_inches="tight", dpi=480)
    plt.close()
    return attribution


def main(args):
    # split configs
    data_cfg: Dict[str, Any] = cv_utils.get_cfg(args.data_cfg)

    # set cuda
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        cv_utils.make_deterministic(args.seed)

    # get dataloader
    print("Building dataset...")
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
    print("Building model...")
    encoder: torch.jit.ScriptModule = torch.jit.load(args.encoder_jit, map_location=device)
    decoder: torch.jit.ScriptModule = torch.jit.load(args.decoder_jit, map_location=device)
    encoder.eval().to(device)
    decoder.eval().to(device)

    def decoder_fn(input: torch.Tensor):
        output = decoder(input)
        return output["pred"]

    ig = captum.attr.IntegratedGradients(decoder_fn)

    attributions: List[np.ndarray] = list()
    img_ids: List[int] = list()
    i = 0
    x: torch.Tensor
    for x, gt in tqdm.tqdm(val_loader, total=args.total):
        i += 1
        if i > args.total:
            break

        x = x.to(device)
        target = gt["label"].to(device)
        img_id = gt["image_id"].item()
        with torch.no_grad():
            seq: torch.Tensor = encoder(x)
        seq.requires_grad_(True)

        attribution = ig.attribute(seq, target=target)

        if args.vis_img:
            img = un_norm(x[0])
            vis_img(img, args.save_path, img_id)
        attribution = vis_attr(
            attribution=attribution,
            save_path=args.save_path,
            img_id=img_id,
            feat_shape=args.feat_shape,
            start_id=args.start_id
        )
        attributions.append(attribution)
        img_ids.append(img_id)

    h5_fp = os.path.join(args.save_path, "attributions.h5")
    with h5py.File(h5_fp, "w") as file:
        for img_id, attribution in zip(img_ids, attributions):
            file[str(img_id)] = attribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--encoder_jit", type=str)
    parser.add_argument("--decoder_jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=10)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--feat_shape", nargs="+", type=int, default=[14, 14])
    parser.add_argument("--vis_img", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
