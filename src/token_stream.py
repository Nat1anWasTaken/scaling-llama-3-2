from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy

from .data import TokenArray
from .tokenizer import Offsets, ShardStore


class TokenStream:
    def __init__(
        self, store: ShardStore, seed: int, rank: int = 0, world_size: int = 1
    ) -> None:
        manifest = store.read_json("manifest.json")

        if manifest is None:
            raise FileNotFoundError(f"no manifest.json in {store.uri}, run prepare.py")

        shards = manifest["shards"]

        if not isinstance(shards, list) or not shards:
            raise ValueError(f"manifest.json in {store.uri} lists no shards")

        self.store = store
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.shard_names = [shard["name"] for shard in shards]

        # The next document to read: which shard, and where in that shard's permutation
        self.shard_index = 0
        self.position = 0
        # Global document counter, so document g belongs to rank g % world_size
        self.documents_seen = 0
        # The RNG as it was before drawing shard_index's permutation
        self.rng_state = numpy.random.default_rng(seed).bit_generator.state

    def state_dict(self) -> dict[str, Any]:
        return {
            "shard_index": self.shard_index,
            "position": self.position,
            "documents_seen": self.documents_seen,
            "rng_state": self.rng_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.shard_index = state["shard_index"]
        self.position = state["position"]
        self.documents_seen = state["documents_seen"]
        self.rng_state = state["rng_state"]

    def __iter__(self) -> Iterator[TokenArray]:
        random_generator = numpy.random.default_rng()
        random_generator.bit_generator.state = self.rng_state
        num_shards = len(self.shard_names)

        # A whole shard (~200MB) is read into memory at once, which over S3 takes a few
        # seconds. Read the next shard on a background thread while this one is consumed;
        # pyarrow releases the GIL during I/O, so the read really overlaps training
        with ThreadPoolExecutor(max_workers=1) as pool:
            if self.shard_index < num_shards:
                pending = pool.submit(
                    self._read_shard, self.shard_names[self.shard_index]
                )

            while self.shard_index < num_shards:
                tokens, offsets = pending.result()

                if self.shard_index + 1 < num_shards:
                    pending = pool.submit(
                        self._read_shard, self.shard_names[self.shard_index + 1]
                    )

                # Every rank draws the same permutations in the same order, so all ranks
                # agree on the global document sequence, and a resumed run redraws it.
                # This is the last step of prepare.py's global shuffle: shards store
                # their documents in fragment order, so it must not be removed
                order = random_generator.permutation(offsets.size - 1)

                while self.position < order.size:
                    document = order[self.position]
                    mine = self.documents_seen % self.world_size == self.rank
                    # Advance before yielding: while suspended, the state means "after
                    # this document"
                    self.position += 1
                    self.documents_seen += 1

                    if mine:
                        yield tokens[offsets[document] : offsets[document + 1]]

                self.shard_index += 1
                self.position = 0
                self.rng_state = random_generator.bit_generator.state

    def _read_shard(self, name: str) -> tuple[TokenArray, Offsets]:
        return self.store.read_tokens(name), self.store.read_offsets(name)
