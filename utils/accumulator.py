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
        self.names = names
        self.accumulators = {n: Accumulator() for n in names}

    def update(self, values: Dict[str, torch.Tensor]):
        for k in self.names:
            self.accumulators[k].update(values[k])

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


class EvalAccumulator:
    ILLEGAL = float("inf")

    def __init__(self, x_len: int, y_len: int):
        self.total = torch.empty(x_len, y_len).fill_(self.ILLEGAL)

    def update(self, x_id: int, y_id: int, value: float):
        self.total[x_id, y_id] = value

    def accumulate(self):
        assert torch.all(self.total != self.ILLEGAL)
        return self.total


class EvalAccumulators:
    def __init__(self, x_len: int, y_len: int, names: List[str]):
        self.accumulators = {n: EvalAccumulator(x_len, y_len) for n in names}
        self.names = names

    def update(self, x_id: int, y_id: int, values: Dict[str, torch.Tensor]):
        for k in self.names:
            self.accumulators[k].update(x_id, y_id, values[k])

    def accumulate(self, to_numpy: bool = True) -> dict:
        ret = dict()
        for k, v in self.accumulators.items():
            value = v.accumulate()
            if to_numpy:
                value = value.detach().numpy()
            ret[k] = value
        return ret

