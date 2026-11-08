from collections.abc import Iterator

import numpy

from .data import TokenArray
from .tokenizer import ShardStore


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

        for name in self.shard_names:
            # A whole shard (~200MB) is read into memory at once. Over S3 this stalls
            # training for a few seconds per shard; prefetching would hide that
            tokens = self.store.read_tokens(name)
            offsets = self.store.read_offsets(name)

            for index in random_generator.permutation(offsets.size - 1):
                yield tokens[offsets[index] : offsets[index + 1]]
