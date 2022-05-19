import os
from typing import Dict, Any, List
import argparse

import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.cuda
import torch.backends.cudnn
import torch.utils.data as data

import cv_lib.utils as cv_utils
from cv_lib.augmentation import UnNormalize

from driver.data import build_eval_dataset
from visualization.vis_feat_attn import vis_img, vis_cnn_feat, vis_vit_feat


def vis_codes(
    origin_feat: torch.Tensor,
    encoded_feat: torch.Tensor,
    codes: torch.LongTensor,
    save_path: str,
    img_id: int,
    vis_channels: int = None,
    norm: bool = True,
    n_sigma: int = 3,
    feat_shape: List[int] = [14, 14],
    start_id: int = 1
):
    save_path = os.path.join(save_path, f"img_{img_id}")
    os.makedirs(save_path, exist_ok=True)
    h5_file = h5py.File(os.path.join(save_path, "feats.h5"), mode="w")

    def vis_feat(feat: torch.Tensor, name: str):
        feat = feat.squeeze()
        fp = os.path.join(save_path, f"{name}.png")
        if feat.dim() == 2:
            feat = vis_vit_feat(
                feat[:, :vis_channels],
                norm=norm,
                n_sigma=n_sigma,
                feat_shape=feat_shape,
                start_id=start_id
            )
        elif feat.dim() == 3:
            feat = vis_cnn_feat(feat[:vis_channels], norm, n_sigma=n_sigma)
        else:
            raise Exception("Invalid dimension")

        feat = feat[0].cpu().numpy()
        h5_file[name] = feat
        sns.heatmap(feat, cmap="vlag", xticklabels=False, yticklabels=False)
        fp = os.path.join(save_path, f"l-{name}.png")
        plt.savefig(fp, bbox_inches="tight", dpi=480)
        plt.close()

    vis_feat(origin_feat, "origin")
    vis_feat(encoded_feat, "encoded")

    codes = codes.squeeze().cpu().numpy()
    h5_file["codes"] = codes
    sns.heatmap(codes, xticklabels=False, yticklabels=False, cbar=False, annot=True, fmt="d", square=True)
    fp = os.path.join(save_path, "codes.png")
    plt.savefig(fp, bbox_inches="tight", dpi=480)
    plt.close()
    h5_file.close()


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
    model: torch.jit.ScriptModule = torch.jit.load(args.jit, map_location="cpu")
    model.eval().to(device)
    with torch.no_grad():
        i = 1
        for x, gt in tqdm.tqdm(val_loader, total=args.total):
            if i > args.total:
                break
            x = x.to(device)
            output: Dict[str, torch.Tensor] = model(x)
            origin_seq = output["origin_seq"]
            encoded_seq = output["encoded_seq"]
            codes = output["match"]

            img_id = gt["image_id"].item()
            if args.vis_img:
                img = un_norm(x[0])
                vis_img(img, args.save_path, img_id)
            vis_codes(
                origin_feat=origin_seq,
                encoded_feat=encoded_seq,
                codes=codes,
                save_path=args.save_path,
                img_id=img_id,
                vis_channels=args.max_channels,
                n_sigma=args.n_sigma,
                feat_shape=args.feat_shape,
                start_id=args.start_id
            )
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=10)
    parser.add_argument("--n_sigma", type=int, default=3)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--feat_shape", nargs="+", type=int, default=[14, 14])
    parser.add_argument("--max_channels", type=int, default=144)
    parser.add_argument("--vis_img", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
