from typing import List

import torch
import torch.nn as nn


@torch.no_grad()
def set_weights(
    model: nn.Module,
    x_direction: List[torch.Tensor],
    y_direction: List[torch.Tensor],
    x_step: float,
    y_step: float
):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    delta = list(dx * x_step + dy * y_step for dx, dy in zip(x_direction, y_direction))
    for param, d in zip(model.parameters(), delta):
        new_value = param + d
        param.copy_(new_value)
