import os
from typing import Dict, Any
import argparse

import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

import torch
import torch.cuda
import torch.backends.cudnn

import cv_lib.utils as cv_utils
from cv_lib.config_parsing import get_cfg

from driver.data import build_train_dataloader
from utils import Accumulators


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
    train_loader, val_loader, _, _ = build_train_dataloader(
        data_cfg,
        num_worker=args.num_worker,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        drop_last=True
    )
    data_loader = train_loader
    if args.use_val_set:
        data_loader = val_loader

    # create model
    print("Building model...")
    model: torch.jit.ScriptModule = torch.jit.load(args.jit, map_location="cpu")
    model.to(device)

    print("Running through dataset")
    batch_idx = torch.arange(args.batch_size, device=device)

    accumulators = Accumulators([
        "corrects",
        f"soft_corrects-T_{args.softmax_temp}",
        "others",
        f"soft_others-T_{args.softmax_temp}"
    ])
    with torch.no_grad():
        model.to(device)
        for x, gt in tqdm.tqdm(data_loader):
            x = x.to(device)
            label = gt["label"].to(device)
            pred, _ = model(x)
            soft_pred = torch.softmax(pred / args.softmax_temp, dim=1)
            tp_mask = torch.zeros_like(pred, dtype=torch.bool)
            tp_mask[batch_idx, label] = True
            accumulators.update({
                "corrects": torch.masked_select(pred, tp_mask),
                f"soft_corrects-T_{args.softmax_temp}": torch.masked_select(soft_pred, tp_mask),
                "others": torch.masked_select(pred, ~tp_mask),
                f"soft_others-T_{args.softmax_temp}": torch.masked_select(soft_pred, ~tp_mask)
            })
    print("Accumulating...")
    accumulators.accumulate()
    values = accumulators.random_select(args.max_number, to_numpy=True)

    print("Ploting...")
    h5_file = h5py.File(os.path.join(args.save_path, "pred.h5"), "w")
    for name, value in values.items():
        h5_file[name] = value

        sns.histplot(data=value, bins=args.bins)
        fp = os.path.join(args.save_path, f"{name}.png")
        plt.savefig(fp, bbox_inches="tight")
        plt.close()
    h5_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str)
    parser.add_argument("--jit", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_val_set", action="store_true")
    parser.add_argument("--max_number", type=int, default=5000)
    parser.add_argument("--softmax_temp", type=int, default=4)
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
