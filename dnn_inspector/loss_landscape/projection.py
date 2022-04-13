from typing import List

import torch


def flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    tensors = list(t.flatten() for t in tensors)
    return torch.cat(tensors)


