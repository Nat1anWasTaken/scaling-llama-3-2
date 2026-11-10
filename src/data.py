# TODO: The DataLoader, produces the Iterable[Batch] that Trainer takes
import itertools
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any, Protocol, TypedDict, cast

import numpy
import torch
from torch import Tensor

# a 1d ndarray with int32
type TokenArray = numpy.ndarray[tuple[int], numpy.dtype[numpy.int32]]

# A row from sequence packer
type PackedRow = tuple[TokenArray, list[int]]


class Stateful(Protocol):
    """
    A pipeline stage whose position can be saved and restored, for resuming a run
    """

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class Batch(TypedDict):
    input_ids: Tensor
    labels: Tensor
    position_ids: Tensor

class RankSharded[T]:
    """
    Keeps every world_size-th item starting at rank, so the ranks split a stream
    without talking to each other
    """

    def __init__(self, items: Iterable[T], rank: int, world_size: int) -> None:
        self.items = items
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[T]:
        return itertools.islice(self.items, self.rank, None, self.world_size)


class SequencePacker:
    """
    把 document 首尾相接，將切成長度剛好是 seq_len 的 row
    """

    def __init__(self, docs: Iterable[TokenArray], seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0")

        self.docs = docs
        self.seq_len = seq_len
        # Tokens not yet emitted in a row, and the document lengths they span.
        # Updated before each yield, so while suspended they mean "after the last row"
        self.buffer: TokenArray = numpy.empty(0, dtype=numpy.int32)
        self.seg_lens: deque[int] = deque()

    def state_dict(self) -> dict[str, Any]:
        return {
            "docs": cast(Stateful, self.docs).state_dict(),
            # A tensor, not an ndarray, so torch.load can keep weights_only=True
            "buffer": torch.from_numpy(self.buffer.copy()),
            "seg_lens": list(self.seg_lens),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cast(Stateful, self.docs).load_state_dict(state["docs"])
        self.buffer = state["buffer"].numpy()
        self.seg_lens = deque(state["seg_lens"])

    def __iter__(self) -> Iterator[PackedRow]:
        for doc in self.docs:
            if doc.dtype != numpy.int32:
                raise TypeError(f"expected int32 tokens, got {doc.dtype}")

            if doc.size == 0:
                continue

            self.buffer = numpy.concatenate([self.buffer, doc])
            self.seg_lens.append(doc.size)

            while self.buffer.size >= self.seq_len:
                row = self.buffer[: self.seq_len]
                self.buffer = self.buffer[self.seq_len :]
                yield row, self._take_row_lens(self.seg_lens)

    def _take_row_lens(self, seg_lens: deque[int]) -> list[int]:
        row_lens: list[int] = []
        need = self.seq_len

        while need > 0:
            n = min(seg_lens[0], need)

            row_lens.append(n)
            need -= n

            if n == seg_lens[0]:
                seg_lens.popleft()
            else:
                seg_lens[0] -= n

        return row_lens


class PackedDataLoader:
    """
    把 Packed Rows 湊成 micro_batch_size 一組，輸出 batch
    """

    def __init__(self, rows: Iterable[PackedRow], micro_batch_size: int) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be greater than 0")

        self.rows = rows
        self.micro_batch_size = micro_batch_size

    # pending is always empty between batches, so the rows hold all the state.
    # The trainer pulls batches synchronously, so this is the state after its last batch
    def state_dict(self) -> dict[str, Any]:
        return cast(Stateful, self.rows).state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        cast(Stateful, self.rows).load_state_dict(state)

    def __iter__(self) -> Iterator[Batch]:
        pending: list[PackedRow] = []

        for row in self.rows:
            pending.append(row)

            if len(pending) == self.micro_batch_size:
                yield self._make_batch(pending)
                pending = []

    @staticmethod
    def _make_batch(rows: list[PackedRow]) -> Batch:
        """
        把 micro_batch_size 個 packed row 組成一個 batch，每個欄位的 shape 是 [B, seq_len]
        """
        input_ids = torch.from_numpy(numpy.stack([row for row, _ in rows])).long()

        position_ids = torch.stack(
            [torch.cat([torch.arange(n) for n in row_lens]) for _, row_lens in rows]
        )

        labels = input_ids.clone()
        labels[position_ids == 0] = -100

        return {"input_ids": input_ids, "labels": labels, "position_ids": position_ids}
