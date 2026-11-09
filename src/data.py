# TODO: The DataLoader, produces the Iterable[Batch] that Trainer takes
import itertools
from collections import deque
from collections.abc import Iterable, Iterator
from typing import TypedDict

import numpy
import torch
from torch import Tensor

# a 1d ndarray with int32
type TokenArray = numpy.ndarray[tuple[int], numpy.dtype[numpy.int32]]

# A row from sequence packer
type PackedRow = tuple[TokenArray, list[int]]


class Batch(TypedDict):
    input_ids: Tensor
    labels: Tensor
    position_ids: Tensor

class RankShardedRow:
    def __init__(self, rows: Iterable[PackedRow], rank: int, world_size: int) -> None:
        self.rows = rows
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[PackedRow]:
        return itertools.islice((self.rows), self.rank, None, self.world_size)


class SequencePacker:
    """
    把 document 首尾相接，將切成長度剛好是 seq_len 的 row
    """

    def __init__(self, docs: Iterable[TokenArray], seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0")

        self.docs = docs
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[PackedRow]:
        buffer: TokenArray = numpy.empty(0, dtype=numpy.int32)
        seg_lens: deque[int] = deque()

        for doc in self.docs:
            if doc.dtype != numpy.int32:
                raise TypeError(f"expected int32 tokens, got {doc.dtype}")

            if doc.size == 0:
                continue

            buffer = numpy.concatenate([buffer, doc])
            seg_lens.append(doc.size)

            start = 0

            while buffer.size - start >= self.seq_len:
                row = buffer[start : start + self.seq_len]
                start += self.seq_len
                yield row, self._take_row_lens(seg_lens)

            buffer = buffer[start:]

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
