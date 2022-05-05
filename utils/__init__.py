from typing import Dict, Tuple

import torch
import torch.nn as nn

from .accumulator import Accumulator, Accumulators


def move_data_to_device(
    x: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = x.to(device)
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            targets[k] = v.to(device)
    return x, targets


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module, is_jit_model: bool = False):
        super().__init__()
        self.module = module
        self.is_jit_model = is_jit_model

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, dict):
            return output
        if self.is_jit_model and isinstance(output, tuple):
            output, _ = output
        ret = {
            "pred": output
        }
        return ret

