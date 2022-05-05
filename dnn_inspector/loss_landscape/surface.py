import collections
import logging
from typing import Dict, Any

import torch


def setup_coordinates(
    coordinates_fp: str,
    coordinate_cfg: Dict[str, Any],
    override: bool = True
) -> Dict[str, torch.Tensor]:
    logger = logging.getLogger("setup_direction")

    def create():
        coordinates = collections.OrderedDict()
        x_coordinate = torch.linspace(**coordinate_cfg["x_axis"])
        coordinates["x_coordinate"] = x_coordinate
        y_coordinate = torch.linspace(**coordinate_cfg["y_axis"])
        coordinates["y_coordinate"] = y_coordinate
        torch.save(coordinates, coordinates_fp)
        logger.info("Coordinates saved")
        return coordinates

    logger.info("Setting up directions...")
    if not override:
        try:
            return torch.load(coordinates_fp)
        except:
            logger.info("Loading surface file failed, creating...")
    return create()
