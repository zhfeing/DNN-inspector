from collections import OrderedDict
import logging
import os
from typing import Dict, Any

import torch


def setup_coordinates(
    coordinates_fp: str,
    coordinate_cfg: Dict[str, Any],
    device: torch.device,
    override: bool = True
):
    logger = logging.getLogger("setup_direction")
    logger.info("Setting up directions")

    if os.path.isfile(coordinates_fp) and not override:
        logger.info("Surface file already exists")
        return

    coordinates = OrderedDict()
    x_coordinate = torch.linspace(device=device, **coordinate_cfg["x_axis"])
    coordinates["x_coordinate"] = x_coordinate
    y_coordinate = torch.linspace(device=device, **coordinate_cfg["y_axis"])
    coordinates["y_coordinate"] = y_coordinate
    torch.save(coordinates, coordinates_fp)
    logger.info("Write coordinates done")
