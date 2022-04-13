from collections import OrderedDict
import logging
import os
from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from .projection import flatten


def setup_direction(
    direction_fp: str,
    model: nn.Module,
    norm: str,
    ignore: str,
    override: bool = True
):
    logger = logging.getLogger("setup_direction")
    logger.info("Setting up directions")

    if os.path.isfile(direction_fp) and not override:
        logger.info("Direction file already exists")
        return

    directions = OrderedDict()
    directions["x_direction"] = create_random_direction(
        model=model,
        norm=norm,
        ignore=ignore
    )
    directions["y_direction"] = create_random_direction(
        model=model,
        norm=norm,
        ignore=ignore
    )
    # sim = torch.cosine_similarity(flatten(directions["x_direction"]), flatten(directions["y_direction"]), dim=0)
    # logger.info("Cosine similarity between x-axis and y-axis is: %s", str(sim))

    torch.save(directions, direction_fp)
    logger.info("Write direction done")


def get_parameters(model: nn.Module) -> List[Tensor]:
    return list(w.detach() for w in model.parameters())


def get_random_weights(weights: List[Tensor]) -> List[Tensor]:
    return list(torch.randn_like(w) for w in weights)


def create_random_direction(
    model: nn.Module,
    norm: str = "filter",
    ignore: str = "bias_bn"
):
    weights = get_parameters(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights, norm, ignore)
    return direction


def normalize_directions_for_weights(
    direction: List[Tensor],
    weights: List[Tensor],
    norm: str,
    ignore: str
):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert len(direction) == len(weights)
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == "bias_bn":
                # set direction to 0
                d.fill_(0)
            else:
                # keep directions for weights/bias that are only 1 per node
                d.copy_(w)
        else:
            normalize_direction(d, w, norm)


def normalize_direction(
    direction: Tensor,
    weights: Tensor,
    norm: str,
    eps: float = 1e-10
):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, "filter" | "layer" | "weight"
    """
    if norm == "filter":
        # Rescale the filters (weights in group) in "direction" so that each
        # filter has the same norm as its corresponding filter in "weights".
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + eps))
    elif norm == "layer":
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm() / direction.norm())
    elif norm == "weight":
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == "dfilter":
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == "dlayer":
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())

