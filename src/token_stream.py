from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy

from .data import TokenArray
from .tokenizer import Offsets, ShardStore


class TokenStream:
    def __init__(self, store: ShardStore, seed: int) -> None:
        manifest = store.read_json("manifest.json")

        if manifest is None:
            raise FileNotFoundError(f"no manifest.json in {store.uri}, run prepare.py")

        shards = manifest["shards"]

        if not isinstance(shards, list) or not shards:
            raise ValueError(f"manifest.json in {store.uri} lists no shards")

        self.store = store
        self.seed = seed
        self.shard_names = [f"{shard['fragment_id']:06d}" for shard in shards]

    def __iter__(self) -> Iterator[TokenArray]:
        random_generator = numpy.random.default_rng(self.seed)

        # A whole shard (~200MB) is read into memory at once, which over S3 takes a few
        # seconds. Read the next shard on a background thread while this one is consumed;
        # pyarrow releases the GIL during I/O, so the read really overlaps training
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self._read_shard, self.shard_names[0])

            for index in range(len(self.shard_names)):
                tokens, offsets = pending.result()

                if index + 1 < len(self.shard_names):
                    pending = pool.submit(self._read_shard, self.shard_names[index + 1])

                # The permutations are drawn in the same order as before, so every rank
                # still sees the same document sequence
                for document in random_generator.permutation(offsets.size - 1):
                    yield tokens[offsets[document] : offsets[document + 1]]

    def _read_shard(self, name: str) -> tuple[TokenArray, Offsets]:
        return self.store.read_tokens(name), self.store.read_offsets(name)
