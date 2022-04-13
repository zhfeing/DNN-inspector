"""
    2D plotting functions
"""
import os
import logging
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

import torch

from loss_landscape.gerenate_vtk import generate_vtp


def plot_contour(
    coordinates_fp: str,
    eval_fp: str,
    save_path: str,
    levels: int,
    vtp_cfg: Dict[str, Any],
    extension: str = "pdf"
):
    logger = logging.getLogger("plot_contour")
    os.makedirs(save_path, exist_ok=True)

    coordinates: Dict[str, torch.Tensor] = torch.load(coordinates_fp, map_location="cpu")
    x_coordinate = coordinates["x_coordinate"].numpy()
    y_coordinate = coordinates["y_coordinate"].numpy()

    eval_ckpt: Dict[str, torch.Tensor] = torch.load(eval_fp, map_location="cpu")
    z_loss: np.ndarray = eval_ckpt["loss"].numpy()
    z_err: np.ndarray = 1 - eval_ckpt["acc"].numpy()

    x_axis, y_axis = np.meshgrid(x_coordinate, y_coordinate)

    logger.info("Loss curve: max: %.5f, min: %.5f", z_loss.max(), z_loss.min())
    logger.info("Error curve: max: %.5f, min: %.5f", z_err.max(), z_err.min())
    assert len(x_coordinate) > 1 and len(y_coordinate) > 1, "x or y coordinates must more than one value"

    # plot 2D contours
    def plot_contour(z: np.ndarray, save_fp: str, levels: int, cmap: str = "RdGy"):
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

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
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

    z_loss = np.log10(z_loss)
    plot_contour(z_err, os.path.join(save_path, "{name}_err." + extension), levels)
    plot_contour(z_loss, os.path.join(save_path, "{name}_log_loss." + extension), levels)

    generate_vtp(
        eval_fp=eval_fp,
        coordinates_fp=coordinates_fp,
        save_fp=os.path.join(save_path, "surface_{name}.vtp"),
        **vtp_cfg
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates_fp")
    parser.add_argument("--eval_fp")
    parser.add_argument("--save_path")
    parser.add_argument("--levels", type=int)
    args = parser.parse_args()

    plot_contour(
        coordinates_fp=args.coordinates_fp,
        eval_fp=args.eval_fp,
        save_path=args.save_path,
        levels=args.levels,
        vtp_cfg=dict()
    )


