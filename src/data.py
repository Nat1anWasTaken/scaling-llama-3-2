# TODO: The DataLoader, produces the Iterable[Batch] that Trainer takes
from typing import TypedDict

from torch import Tensor


class Batch(TypedDict):
    input_ids: Tensor
    labels: Tensor
    position_ids: Tensor
