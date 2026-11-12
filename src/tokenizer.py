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

    def read_bucket_offsets(self, name: str) -> Offsets:
        return self._load(self._path(name, "bucket_offsets.npy"))

    def read_offsets_slice(self, name: str, start: int, stop: int) -> Offsets:
        return self._load_slice(self._path(name, "offsets.npy"), start, stop)

    def read_tokens_slice(self, name: str, start: int, stop: int) -> TokenArray:
        return self._load_slice(self._path(name, "tokens.npy"), start, stop)

    def write_shard(
        self,
        name: str,
        tokens: TokenArray,
        offsets: Offsets,
        bucket_offsets: Offsets | None = None,
    ) -> None:
        self.filesystem.create_dir(str(self._path(name)), recursive=True)
        self._save(self._path(name, "tokens.npy"), tokens)

        if bucket_offsets is not None:
            self._save(self._path(name, "bucket_offsets.npy"), bucket_offsets)

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

    def _load_slice(self, path: PurePosixPath, start: int, stop: int) -> numpy.ndarray:
        """
        Elements [start, stop) of a 1d .npy file, without reading the rest. On S3 the
        seek + read becomes a range request
        """
        with self.filesystem.open_input_file(str(path)) as file:
            version = numpy.lib.format.read_magic(file)

            if version == (1, 0):
                _, _, dtype = numpy.lib.format.read_array_header_1_0(file)
            else:
                _, _, dtype = numpy.lib.format.read_array_header_2_0(file)

            # The data starts right after the header
            file.seek(file.tell() + start * dtype.itemsize)
            data = file.read((stop - start) * dtype.itemsize)

        return numpy.frombuffer(data, dtype=dtype)


def fragment_shard_name(fragment_id: int) -> str:
    """
    Where phase 1 of the global shuffle puts a tokenized fragment, apart from the
    output shards that TokenStream reads
    """
    return f"fragments/{fragment_id:06d}"


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
    """
    Phase 1 of the global shuffle: tokenizes a fragment and scatters its documents into
    buckets. Documents are stored sorted by bucket, so each bucket is one contiguous
    range that phase 2 can read on its own
    """

    def __init__(
        self,
        dataset: LanceDataset,
        tokenizer: PreTrainedTokenizerFast,
        exclude_ids: frozenset[str],
        store: ShardStore,
        seed: int,
        num_buckets: int,
        batch_size: int = 1024,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if num_buckets <= 0:
            raise ValueError("num_buckets must be greater than 0")

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.exclude_ids = exclude_ids
        self.store = store
        self.seed = seed
        self.num_buckets = num_buckets
        self.batch_size = batch_size

    def tokenize_fragment(self, fragment_id: int) -> tuple[int, Offsets]:
        """
        Returns the fragment's token count and its bucket_offsets: bucket b holds
        documents [bucket_offsets[b], bucket_offsets[b + 1])
        """
        name = fragment_shard_name(fragment_id)

        if self.store.has_shard(name):
            return (
                int(self.store.read_offsets(name)[-1]),
                self.store.read_bucket_offsets(name),
            )

        documents: list[TokenArray] = []

        for texts in self._iter_texts(fragment_id):
            documents.extend(self._encode(texts))

        # Every document draws its bucket independently and uniformly. The draws depend
        # only on (seed, fragment_id), not on which worker runs it or when
        random_generator = numpy.random.default_rng([self.seed, 0, fragment_id])
        buckets = random_generator.integers(self.num_buckets, size=len(documents))

        order = numpy.argsort(buckets, kind="stable")
        documents = [documents[i] for i in order]

        bucket_offsets = numpy.zeros(self.num_buckets + 1, dtype=numpy.int64)
        numpy.cumsum(
            numpy.bincount(buckets, minlength=self.num_buckets),
            out=bucket_offsets[1:],
        )

        tokens, offsets = self._concatenate(documents)

        self.store.write_shard(name, tokens, offsets, bucket_offsets)
        return int(offsets[-1]), bucket_offsets

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


class BucketGatherer:
    """
    Phase 2 of the global shuffle: collects one bucket's documents from every fragment
    into one output shard. The documents stay in fragment order inside the shard;
    TokenStream's per-shard permutation is the shuffle's last step
    """

    def __init__(
        self, store: ShardStore, fragment_names: list[str], bucket_offsets: Offsets
    ) -> None:
        # bucket_offsets has shape [num_fragments, num_buckets + 1]
        if bucket_offsets.shape[0] != len(fragment_names):
            raise ValueError("need one row of bucket_offsets per fragment")

        self.store = store
        self.fragment_names = fragment_names
        self.bucket_offsets = bucket_offsets

    def gather_bucket(self, bucket: int) -> int:
        name = f"{bucket:06d}"

        if self.store.has_shard(name):
            return int(self.store.read_offsets(name)[-1])

        token_parts: list[TokenArray] = []
        length_parts: list[Offsets] = []

        for fragment_name, row in zip(
            self.fragment_names, self.bucket_offsets, strict=True
        ):
            start, stop = int(row[bucket]), int(row[bucket + 1])

            if start == stop:
                continue

            # stop - start documents are bounded by stop - start + 1 offsets
            offsets = self.store.read_offsets_slice(fragment_name, start, stop + 1)
            token_parts.append(
                self.store.read_tokens_slice(
                    fragment_name, int(offsets[0]), int(offsets[-1])
                )
            )
            length_parts.append(numpy.diff(offsets))

        if token_parts:
            tokens = numpy.concatenate(token_parts)
            lengths = numpy.concatenate(length_parts)
        else:
            tokens = numpy.empty(0, dtype=numpy.int32)
            lengths = numpy.empty(0, dtype=numpy.int64)

        offsets = numpy.zeros(lengths.size + 1, dtype=numpy.int64)
        numpy.cumsum(lengths, out=offsets[1:])

        self.store.write_shard(name, tokens, offsets)
        return int(offsets[-1])
