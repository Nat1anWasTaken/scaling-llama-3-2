from collections.abc import Iterator
from pathlib import Path

import lance
import numpy
import pyarrow.parquet
from huggingface_hub import snapshot_download
from lance import LanceDataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .data import TokenArray


def load_tokenizer(name: str) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(name)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"expected a PreTrainedTokenizerFast tokenizer, got {type(tokenizer).__name__}"
        )

    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must have bos_token_id and eos_token_id")

    return tokenizer


def load_development_ids(repository_id: str, expected_size: int) -> frozenset[str]:
    local_dir = Path(
        snapshot_download(
            repository_id, repo_type="dataset", allow_patterns=["*.parquet"]
        )
    )

    document_ids: set[str] = set()

    for path in sorted(local_dir.rglob("*.parquet")):
        table = pyarrow.parquet.read_table(path, columns=["id"])
        document_ids.update(table.column("id").to_pylist())

    # An empty or partial set would silently leak eval documents into training
    if len(document_ids) != expected_size:
        raise ValueError(
            f"expected {expected_size} development ids, got {len(document_ids)}"
        )

    return frozenset(document_ids)


def open_dataset(uri: str) -> LanceDataset:
    return lance.dataset(uri)


def shuffled_fragment_ids(dataset: LanceDataset, seed: int) -> list[int]:
    """
    Return the dataset's fragment ids in a random order, fixed by seed.
    """
    # Shuffle the real ids: Lance doesn't guarantee they are contiguous 0..N-1.
    # Sort first so the result depends only on the seed, not on listing order
    fragment_ids = sorted(fragment.fragment_id for fragment in dataset.get_fragments())
    random_generator = numpy.random.default_rng(seed)
    return random_generator.permutation(fragment_ids).tolist()


class FragmentTokenizer:
    def __init__(
        self,
        dataset: LanceDataset,
        tokenizer: PreTrainedTokenizerFast,
        exclude_ids: frozenset[str],
        output_root: Path,
        batch_size: int = 1024,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.exclude_ids = exclude_ids
        self.output_root = output_root
        self.batch_size = batch_size

    def tokenize_fragment(self, fragment_id: int) -> int:
        output_dir = self.output_root / f"{fragment_id:06d}"

        if output_dir.exists():
            return int(numpy.load(output_dir / "offsets.npy")[-1])

        documents: list[TokenArray] = []

        for texts in self._iter_texts(fragment_id):
            documents.extend(self._encode(texts))

        self._write_shard(output_dir, documents)
        return sum(document.size for document in documents)

    def _iter_texts(self, fragment_id: int) -> Iterator[list[str]]:
        fragment = self.dataset.get_fragment(fragment_id)

        if fragment is None:
            raise ValueError(f"fragment {fragment_id} does not exist")

        for batch in fragment.to_batches(
            columns=["id", "text"], batch_size=self.batch_size
        ):
            document_ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()

            yield [
                text
                for document_id, text in zip(document_ids, texts, strict=True)
                if text and document_id not in self.exclude_ids
            ]

    def _encode(self, texts: list[str]) -> list[TokenArray]:
        # add_special_tokens=False because we will add them later
        encoded = self.tokenizer(
            texts, add_special_tokens=False, return_attention_mask=False
        )

        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        return [
            numpy.array([bos_token_id, *token_ids, eos_token_id], dtype=numpy.int32)
            for token_ids in encoded["input_ids"]
        ]

    @staticmethod
    def _write_shard(output_dir: Path, documents: list[TokenArray]) -> None:
        lengths = numpy.fromiter(
            (document.size for document in documents),
            dtype=numpy.int64,
            count=len(documents)
        )

        offsets = numpy.zeros(len(documents) + 1, dtype=numpy.int64)

        numpy.cumsum(lengths, out=offsets[1:])

        if documents:
            tokens = numpy.concatenate(documents)
        else:
            tokens = numpy.empty(0, dtype=numpy.int32)

        temporary_dir = output_dir.with_name(output_dir.name + ".tmp")
        temporary_dir.mkdir(parents=True, exist_ok=True)

        numpy.save(temporary_dir / "tokens.npy", tokens)
        numpy.save(temporary_dir / "offsets.npy", offsets)

        # Since rename is atomic, output_dir exists = shard is complete
        temporary_dir.rename(output_dir)
