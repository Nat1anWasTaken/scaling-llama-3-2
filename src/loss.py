from typing import Protocol

import torch.nn.functional
from torch import Tensor

IGNORE_INDEX = -100


class CausalLMLossSum(Protocol):
    def __call__(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class CrossEntropyLossSum:
    def __call__(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        # Upcast: a bf16 log-softmax over a 128k vocabulary loses too much precision
        loss_sum = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)).float(),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        num_tokens = (shifted_labels != IGNORE_INDEX).sum()

        return loss_sum, num_tokens
