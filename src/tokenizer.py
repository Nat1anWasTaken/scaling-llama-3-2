import io
import json
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import lance
import numpy
import pyarrow.fs
import pyarrow.parquet
from huggingface_hub import snapshot_download
from lance import LanceDataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .data import TokenArray

type Offsets = numpy.ndarray[tuple[int], numpy.dtype[numpy.int64]]


class ShardStore:
    def __init__(self, uri: str) -> None:
        if "://" not in uri:
            uri = str(Path(uri).absolute())

        self.uri = uri

        filesystem, root = pyarrow.fs.FileSystem.from_uri(uri)

        self.filesystem: pyarrow.fs.FileSystem = filesystem
        self.root = PurePosixPath(root)

    def has_shard(self, name: str) -> bool:
        return self._is_file(self._path(name, "offsets.npy"))

    def read_offsets(self, name: str) -> Offsets:
        return self._load(self._path(name, "offsets.npy"))

    def read_tokens(self, name: str) -> TokenArray:
        return self._load(self._path(name, "tokens.npy"))

    def write_shard(self, name: str, tokens: TokenArray, offsets: Offsets) -> None:
        self.filesystem.create_dir(str(self._path(name)), recursive=True)
        self._save(self._path(name, "tokens.npy"), tokens)

        # Commit point: offsets.npy only appears once it is fully written
        temporary_path = self._path(name, "offsets.npy.tmp")
        self._save(temporary_path, offsets)
        self.filesystem.move(str(temporary_path), str(self._path(name, "offsets.npy")))

    def read_json(self, name: str) -> dict[str, object] | None:
        path = self._path(name)

        if not self._is_file(path):
            return None

        with self.filesystem.open_input_stream(str(path)) as stream:
            return json.loads(stream.read())

    def write_json(self, name: str, content: dict[str, object]) -> None:
        self.filesystem.create_dir(str(self.root), recursive=True)

        temporary_path = self._path(name + ".tmp")

        with self.filesystem.open_output_stream(str(temporary_path)) as stream:
            stream.write(json.dumps(content, indent=2).encode())

        self.filesystem.move(str(temporary_path), str(self._path(name)))

    # Paths stay PurePosixPath inside the class; the damn pyarrow only accepts str, so
    # convert with str() right at each pyarrow call

    def _path(self, *parts: str) -> PurePosixPath:
        return self.root.joinpath(*parts)

    def _is_file(self, path: PurePosixPath) -> bool:
        # A missing path doesn't raise, it reports FileType.NotFound
        info = self.filesystem.get_file_info(str(path))
        return info.type == pyarrow.fs.FileType.File

    def _save(self, path: PurePosixPath, array: numpy.ndarray) -> None:
        with self.filesystem.open_output_stream(str(path)) as stream:
            numpy.save(stream, array)

    def _load(self, path: PurePosixPath) -> numpy.ndarray:
        # open_input_file, not open_input_stream: numpy.load needs to seek
        with self.filesystem.open_input_file(str(path)) as file:
            data = file.read()

        return numpy.load(io.BytesIO(data), allow_pickle=False)


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

    if len(document_ids) != expected_size:
        raise ValueError(
            f"expected {expected_size} development ids, got {len(document_ids)}"
        )

    return frozenset(document_ids)


def open_dataset(uri: str) -> LanceDataset:
    return lance.dataset(uri)


def shuffled_fragment_ids(dataset: LanceDataset, seed: int) -> list[int]:
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
        store: ShardStore,
        batch_size: int = 1024,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.exclude_ids = exclude_ids
        self.store = store
        self.batch_size = batch_size

    def tokenize_fragment(self, fragment_id: int) -> int:
        name = f"{fragment_id:06d}"

        if self.store.has_shard(name):
            return int(self.store.read_offsets(name)[-1])

        documents: list[TokenArray] = []

        for texts in self._iter_texts(fragment_id):
            documents.extend(self._encode(texts))

        tokens, offsets = self._concatenate(documents)

        self.store.write_shard(name, tokens, offsets)
        return sum(document.size for document in documents)

    @staticmethod
    def _concatenate(documents: list[TokenArray]) -> tuple[TokenArray, Offsets]:
        lengths = numpy.fromiter(
            (document.size for document in documents),
            dtype=numpy.int64,
            count=len(documents),
        )

        offsets = numpy.zeros(len(documents) + 1, dtype=numpy.int64)

        numpy.cumsum(lengths, out=offsets[1:])

        if documents:
            tokens = numpy.concatenate(documents)
        else:
            tokens = numpy.empty(0, dtype=numpy.int32)

        return tokens, offsets

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
