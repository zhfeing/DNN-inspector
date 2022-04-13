from typing import Dict, Tuple

import torch


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
