import os
from typing import Dict, Any, Tuple, List
import argparse

import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

import torch
import torch.cuda
import torch.backends.cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.data as data

import cv_lib.utils as cv_utils
from cv_lib.config_parsing import get_cfg
import cv_lib.visualization as vis
from cv_lib.augmentation import UnNormalize

from driver.data import build_eval_dataset


def vis_img(img: torch.Tensor, save_path: str, img_id: int):
    fp = os.path.join(save_path, f"raw_img_{img_id}.png")
    img = TF.to_pil_image(img)
    img.save(fp)


def vis_cnn_feat(feat: torch.Tensor, norm: bool = True, n_sigma: int = 3) -> torch.Tensor:
    if norm:
        # [C, H, W]
        shape = feat.shape
        feat = feat.flatten(1)
        feat.t_()
        feat = F.layer_norm(feat, (shape[0],))
        feat.t_()
        feat = feat.unflatten(-1, shape[1:])
    feat = vis.vis_featuremap(
        feat=feat,
        n_sigma=n_sigma,
        padding=1
    )
    return feat


def vis_vit_feat(
    feat: torch.Tensor,
    feat_shape: Tuple[int, int],
    norm: bool = True,
    start_id: int = 1,
    n_sigma: int = 3
):
    if norm:
        # [N, dim]
        feat = F.layer_norm(feat, (feat.shape[1],))
    feat = vis.vis_seq_token(
        seq=feat[start_id:],
        feat_shape=feat_shape,
        n_sigma=n_sigma,
        padding=1
    )
    return feat


def vis_mid_feat(
    features: Dict[str, torch.Tensor],
    save_path: str,
    img_id: int,
    vis_channels: int = None,
    norm: bool = True,
    n_sigma: int = 3,
    feat_shape: List[int] = [14, 14],
    start_id: int = 1
):
    save_path = os.path.join(save_path, f"seq_img_{img_id}")
    os.makedirs(save_path, exist_ok=True)
    h5_file = h5py.File(os.path.join(save_path, "feats.h5"), mode="w")
    for name, val in features.items():
        val = val.squeeze()
        fp = os.path.join(save_path, f"l-{name}.png")
        if val.dim() == 2:
            feat = vis_vit_feat(
                val[:, :vis_channels],
                norm=norm,
                n_sigma=n_sigma,
                feat_shape=feat_shape,
                start_id=start_id
            )
        elif val.dim() == 3:
            feat = vis_cnn_feat(val[:vis_channels], norm, n_sigma=n_sigma)
        else:
            raise Exception("Invalid dimension")

        feat = feat[0].cpu().numpy()
        h5_file[name] = feat
        sns.heatmap(feat, cmap="vlag", xticklabels=False, yticklabels=False)
        fp = os.path.join(save_path, f"l-{name}.png")
        plt.savefig(fp, bbox_inches="tight")
        plt.close()
    h5_file.close()


def vis_attn(attentions: Dict[str, torch.Tensor], save_path: str, img_id: int):
    save_path = os.path.join(save_path, f"seq_img_{img_id}")
    os.makedirs(save_path, exist_ok=True)
    h5_file = h5py.File(os.path.join(save_path, "attn.h5"), mode="w")
    for name, val in attentions.items():
        val = val.squeeze()
        fp = os.path.join(save_path, f"l-{name}.png")
        assert val.dim() == 3
        # val = torch.pow(val, 0.5)
        feat = val.mean(dim=0).cpu().numpy()
        attn = feat[1:, 1:]
        h5_file[name] = attn
        sns.heatmap(attn, cmap="coolwarm", xticklabels=False, yticklabels=False)
        fp = os.path.join(save_path, f"l-{name}-attn.png")
        plt.savefig(fp, bbox_inches="tight")
        plt.close()
    h5_file.close()


def main(args):
    # split configs
    data_cfg: Dict[str, Any] = get_cfg(args.data_cfg)

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
    model.to(device)
    with torch.no_grad():
        model.to(device)
        i = 1
        for x, gt in tqdm.tqdm(val_loader, total=args.total):
            if i > args.total:
                break
            x = x.to(device)
            _, feats = model(x)

            mid_feat = {n: feats[n] for n in args.feat_names}
            mid_attn = {n: feats[n] for n in args.attn_names}

            img_id = gt["image_id"].item()
            if args.vis_img:
                img = un_norm(x[0])
                vis_img(img, args.save_path, img_id)
            if args.vis_feat and len(mid_feat) > 0:
                vis_mid_feat(
                    mid_feat,
                    args.save_path,
                    img_id,
                    n_sigma=args.n_sigma,
                    feat_shape=args.feat_shape,
                    start_id=args.start_id
                )
            if args.vis_attn and len(mid_attn) > 0:
                vis_attn(mid_attn, args.save_path, img_id)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--total", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_sigma", type=int, default=3)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--feat_names", nargs="+", default=list())
    parser.add_argument("--attn_names", nargs="+", default=list())
    parser.add_argument("--feat_shape", nargs="+", type=int)
    parser.add_argument("--vis_img", action="store_true")
    parser.add_argument("--vis_feat", action="store_true")
    parser.add_argument("--vis_attn", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
