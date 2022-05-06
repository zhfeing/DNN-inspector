import collections
import logging
from typing import List, Dict

import torch
import torch.nn as nn


def setup_direction(
    direction_fp: str,
    model: nn.Module,
    norm: str,
    ignore: str,
    override: bool = True
) -> Dict[str, torch.Tensor]:
    logger = logging.getLogger("setup_direction")
    logger.info("Setting up directions")

    def create():
        directions = collections.OrderedDict()
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
        torch.save(directions, direction_fp)
        logger.info("Direction saved")
        return directions

    if not override:
        try:
            return torch.load(direction_fp)
        except:
            logger.info("Loading surface file failed, creating...")

    directions = create()
    sim = torch.cosine_similarity(
        flatten(directions["x_direction"]),
        flatten(directions["y_direction"]),
        dim=0
    )
    logger.info("Cosine similarity between x-axis and y-axis is: %s", str(sim))
    return directions


@torch.no_grad()
def get_parameters(model: nn.Module) -> List[torch.Tensor]:
    return list(model.parameters())


def get_random_weights(weights: List[torch.Tensor]) -> List[torch.Tensor]:
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


@torch.no_grad()
def normalize_directions_for_weights(
    direction: List[torch.Tensor],
    weights: List[torch.Tensor],
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


@torch.no_grad()
def normalize_direction(
    direction: torch.Tensor,
    weights: torch.Tensor,
    norm: str,
    eps: float = 1e-7
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
        direction.mul_(weights.norm(dim=0, keepdim=True) / (direction.norm(dim=0, keepdim=True) + eps))
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
            d.div_(d.norm() + eps)
    elif norm == "dlayer":
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())
    else:
        raise NotImplementedError


def flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    tensors = list(t.flatten() for t in tensors)
    return torch.cat(tensors)

