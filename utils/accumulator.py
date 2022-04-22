from typing import List, Dict

import torch


class Accumulator:
    def __init__(self):
        self.values = list()

    def update(self, value: torch.Tensor):
        self.values.append(value.cpu())

    def accumulate(self):
        self.values = torch.cat(self.values)

    def random_select(self, max_number) -> torch.Tensor:
        pick = torch.randperm(self.values.shape[0])[:max_number]
        return self.values[pick]


class Accumulators:
    def __init__(self, names: List[str]):
        self.accumulators = {n: Accumulator() for n in names}

    def update(self, values: Dict[str, torch.Tensor]):
        for k, v in values.items():
            self.accumulators[k].update(v)

    def accumulate(self):
        for v in self.accumulators.values():
            v.accumulate()

    def random_select(self, max_number, to_numpy: bool = True) -> dict:
        ret = dict()
        for k, v in self.accumulators.items():
            v = v.random_select(max_number)
            if to_numpy:
                ret[k] = v.numpy()
        return ret
