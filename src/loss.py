from typing import Protocol

from torch import Tensor


class CausalLMLossSum(Protocol):
    def __call__(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
