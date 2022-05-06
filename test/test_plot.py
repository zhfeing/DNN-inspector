import argparse
import itertools
import os
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np

import torch

from dnn_inspector.loss_landscape.gerenate_vtk import generate_vtp
from utils.accumulator import EvalAccumulators


def plot_landscape(
    save_path: str,
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
    transformers: Dict[str, Callable[[np.ndarray], np.ndarray]],
    values: Dict[str, np.ndarray],
    levels: int = 10,
    # vtp_cfg: Dict[str, Any],
):
    x_axis, y_axis = np.meshgrid(x_coordinate, y_coordinate)
    assert len(x_coordinate) > 1 and len(y_coordinate) > 1, "x or y coordinates must more than one value"

    # plot 2D contours
    def plot_value(z: np.ndarray, save_fp: str, levels: int, cmap: str = "RdGy"):
        contour = plt.contour(
            x_axis, y_axis, z,
            cmap=cmap,
            levels=levels
        )
        plt.clabel(contour, inline=True, fontsize=8)
        plt.savefig(
            save_fp.format(name="contour"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        plt.contourf(
            x_axis, y_axis, z,
            cmap=cmap,
            levels=levels
        )
        plt.colorbar()
        plt.savefig(
            save_fp.format(name="contourf"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        _, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(
            x_axis, y_axis, z,
            cmap=cmap,
            linewidth=0,
            antialiased=False
        )
        plt.savefig(
            save_fp.format(name="surface"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    for k, v in values.items():
        value = transformers[k](v)
        print("ploting", k)
        plot_value(value, os.path.join(save_path, "{name}_" + f"{k}.png"), levels)

    # generate_vtp(
    #     eval_fp=eval_fp,
    #     coordinates_fp=coordinates_fp,
    #     save_fp=os.path.join(save_path, "surface_{name}.vtp"),
    #     **vtp_cfg
    # )


def identity(x):
    return x


def norm(x: np.ndarray):
    return x / x.max()


def main(args):
    coordinates_fp = os.path.join(args.save_path, "coordinates.pth")
    coordinates = torch.load(coordinates_fp, map_location="cpu")

    transformers = {
        "loss": norm,
        "err@1": identity,
        "model_l2": norm,
        "loss+l2": norm
    }
    l2_weight = 0.05
    x_len = len(coordinates["x_coordinate"])
    y_len = len(coordinates["y_coordinate"])
    accumulators = EvalAccumulators(x_len, y_len, transformers.keys())
    iters = itertools.product(range(x_len), range(y_len))
    for task_id, (x_id, y_id) in enumerate(iters):
        fp = os.path.join(args.save_path, "eval", f"{task_id}.pkl")
        values = torch.load(fp, map_location="cpu")
        values["err@1"] = 100 * (1 - values["acc"][1])
        values["loss+l2"] = values["loss"] + l2_weight * values["model_l2"]
        accumulators.update(x_id, y_id, values)
    values: Dict[str, np.ndarray] = accumulators.accumulate(to_numpy=True)
    plot_landscape(
        save_path=args.save_path,
        x_coordinate=coordinates["x_coordinate"].numpy(),
        y_coordinate=coordinates["y_coordinate"].numpy(),
        transformers=transformers,
        values=values,
        # levels=5
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "eval"), exist_ok=True)
    main(args)
