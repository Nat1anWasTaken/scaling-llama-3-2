from typing import Protocol

import torch.nn.functional
from torch import Tensor

IGNORE_INDEX = -100


class CausalLMLossSum(Protocol):
    def __call__(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class CrossEntropyLossSum:
    def __call__(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        # Position t predicts token t + 1. Shift the labels left instead of slicing the
        # logits: logits[:, :-1] isn't contiguous, so reshaping it copied all the logits,
        # while the full logits reshape into a view. The last position gets no target
        shifted_labels = torch.nn.functional.pad(
            labels[:, 1:], (0, 1), value=IGNORE_INDEX
        )

        # Upcast: a bf16 log-softmax over a 128k vocabulary loses too much precision
        loss_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            shifted_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        num_tokens = (shifted_labels != IGNORE_INDEX).sum()

        return loss_sum, num_tokens
