"""
Pre-tokenize FineWeb-Edu into globally shuffled shards, until a token budget.

    eval "$(flock s3 credentials --ttl 24h)"
    uv run prepare.py --output s3://<bucket>/fineweb-edu-llama3 --token-budget 30000000000 \\
        --run-name prepare-fineweb-edu-llama3

A two-pass shuffle:
  1. Tokenize fragments in a fixed shuffled order until the budget. Every document draws
     a bucket uniformly at random, and each fragment is stored sorted by bucket under
     fragments/.
  2. Gather each bucket's documents from every fragment into one output shard.
TokenStream then permutes the documents inside each shard. Uniform buckets plus a
uniform permutation inside each bucket make the whole document order uniformly random,
so any prefix of the shards is a uniform sample of the data.

Rerunning with the same --output skips finished fragments and shards, so an interrupted
run (or an expired S3 key) only needs a rerun. The bucket count comes from the budget,
so a different budget needs a different --output. fragments/ can be deleted once
manifest.json is written.
"""

import argparse
import math
import multiprocessing
import os
import time

import numpy

from src.loggers import ConsoleLogger, WandbLogger
from src.tokenizer import (
    BucketGatherer,
    FragmentTokenizer,
    Offsets,
    ShardStore,
    fragment_shard_name,
    load_development_ids,
    load_tokenizer,
    open_dataset,
    shuffled_fragment_ids,
)

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
DATASET_URI = "s3://flock-datasets/fineweb-edu"
DEVELOPMENT_DATASET = "cerulean-works/fineweb-edu-dev"
DEVELOPMENT_SIZE = 5000
SHARD_FORMAT = (
    "[BOS, text, EOS] int32 tokens.npy + int64 offsets.npy, "
    "globally shuffled into buckets"
)
WANDB_PROJECT = "lab6-predictable-scaling"

# One per worker process, built by initialize_tokenizer or initialize_gatherer
tokenizer_worker: FragmentTokenizer | None = None
gatherer_worker: BucketGatherer | None = None


def initialize_tokenizer(
    dataset_uri: str,
    tokenizer_name: str,
    exclude_ids: frozenset[str],
    output_uri: str,
    seed: int,
    num_buckets: int,
    batch_size: int,
) -> None:
    global tokenizer_worker

    # Each process already uses one core; tokenizer threads on top would oversubscribe
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer_worker = FragmentTokenizer(
        open_dataset(dataset_uri),
        load_tokenizer(tokenizer_name),
        exclude_ids,
        ShardStore(output_uri),
        seed,
        num_buckets,
        batch_size,
    )


def tokenize_fragment(fragment_id: int) -> tuple[int, int, Offsets]:
    if tokenizer_worker is None:
        raise RuntimeError("worker is not initialized")

    return fragment_id, *tokenizer_worker.tokenize_fragment(fragment_id)


def initialize_gatherer(
    output_uri: str, fragment_names: list[str], bucket_offsets: Offsets
) -> None:
    global gatherer_worker

    gatherer_worker = BucketGatherer(
        ShardStore(output_uri), fragment_names, bucket_offsets
    )


def gather_bucket(bucket: int) -> tuple[int, int]:
    if gatherer_worker is None:
        raise RuntimeError("worker is not initialized")

    return bucket, gatherer_worker.gather_bucket(bucket)


def check_metadata(store: ShardStore, metadata: dict[str, object]) -> None:
    """
    Shards are skipped by name only, so refuse to mix shards made with different settings
    """
    existing = store.read_json("metadata.json")

    if existing is None:
        store.write_json("metadata.json", metadata)
    elif existing != metadata:
        raise ValueError(
            f"{store.uri} holds shards made with {existing}, "
            f"not {metadata}; use a different --output"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", required=True, help="local path or s3://bucket/prefix"
    )
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=50_000_000,
        help="target tokens per output shard; sets the bucket count from the budget",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--dataset", default=DATASET_URI)
    parser.add_argument("--development-dataset", default=DEVELOPMENT_DATASET)
    parser.add_argument("--run-name", required=True, help="names the wandb run")
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    store = ShardStore(arguments.output)
    dataset = open_dataset(arguments.dataset)

    num_buckets = math.ceil(arguments.token_budget / arguments.shard_tokens)

    # The fragments' bucket assignment depends on seed and num_buckets, so both must
    # match for a rerun to reuse them
    metadata = {
        "tokenizer": arguments.tokenizer,
        "dataset": arguments.dataset,
        "dataset_version": dataset.version,
        "development_dataset": arguments.development_dataset,
        "format": SHARD_FORMAT,
        "seed": arguments.seed,
        "num_buckets": num_buckets,
    }
    check_metadata(store, metadata)

    if arguments.no_wandb:
        logger: ConsoleLogger | WandbLogger = ConsoleLogger()
    else:
        logger = WandbLogger(
            WANDB_PROJECT, arguments.run_name, {**vars(arguments), **metadata}
        )

    # Download into the HF cache once here, so workers don't all race to download it,
    load_tokenizer(arguments.tokenizer)

    exclude_ids = load_development_ids(arguments.development_dataset, DEVELOPMENT_SIZE)
    fragment_ids = shuffled_fragment_ids(dataset, arguments.seed)

    fragments: list[dict[str, int]] = []
    bucket_offsets: list[Offsets] = []
    total_tokens = 0
    start_time = time.monotonic()

    # spawn, not fork: this process already runs Lance's background threads, and
    # forking a process with live threads can deadlock the child
    context = multiprocessing.get_context("spawn")

    # Phase 1: tokenize and scatter
    with context.Pool(
        arguments.workers,
        initializer=initialize_tokenizer,
        initargs=(
            arguments.dataset,
            arguments.tokenizer,
            exclude_ids,
            store.uri,
            arguments.seed,
            num_buckets,
            arguments.batch_size,
        ),
    ) as pool:
        # imap yields in fragment_ids order however the workers finish, so the
        # selected fragments are always the same prefix of the shuffled order
        for fragment_id, tokens, offsets in pool.imap(tokenize_fragment, fragment_ids):
            fragments.append({"fragment_id": fragment_id, "tokens": tokens})
            bucket_offsets.append(offsets)
            total_tokens += tokens

            elapsed = time.monotonic() - start_time
            logger.log(
                {
                    "prepare/fragment_id": fragment_id,
                    "prepare/fragment_tokens": tokens,
                    "prepare/total_tokens": total_tokens,
                    "prepare/budget_fraction": total_tokens / arguments.token_budget,
                    "prepare/tokens_per_second": total_tokens / elapsed,
                },
                step=len(fragments),
            )

            if total_tokens >= arguments.token_budget:
                break

        # Leaving the with block terminates workers still busy on later fragments.
        # Their shards never got offsets.npy, so a rerun redoes them

    if total_tokens < arguments.token_budget:
        print(f"warning: dataset ran out at {total_tokens} tokens", flush=True)

    shards: list[dict[str, object]] = []
    gathered_tokens = 0
    start_time = time.monotonic()

    # Phase 2: gather each bucket into an output shard
    with context.Pool(
        arguments.workers,
        initializer=initialize_gatherer,
        initargs=(
            store.uri,
            [fragment_shard_name(fragment["fragment_id"]) for fragment in fragments],
            numpy.stack(bucket_offsets),
        ),
    ) as pool:
        for bucket, tokens in pool.imap(gather_bucket, range(num_buckets)):
            shards.append({"name": f"{bucket:06d}", "tokens": tokens})
            gathered_tokens += tokens

            elapsed = time.monotonic() - start_time
            logger.log(
                {
                    "prepare/shard_tokens": tokens,
                    "prepare/gathered_tokens": gathered_tokens,
                    "prepare/gathered_fraction": gathered_tokens / total_tokens,
                    "prepare/gather_tokens_per_second": gathered_tokens / elapsed,
                },
                step=len(fragments) + len(shards),
            )

    # Every document lands in exactly one bucket
    if gathered_tokens != total_tokens:
        raise RuntimeError(
            f"gathered {gathered_tokens} tokens but tokenized {total_tokens}"
        )

    store.write_json(
        "manifest.json",
        {
            "seed": arguments.seed,
            "token_budget": arguments.token_budget,
            "num_buckets": num_buckets,
            "total_tokens": total_tokens,
            "fragments": fragments,
            "shards": shards,
        },
    )

    if isinstance(logger, WandbLogger):
        logger.run.summary["total_tokens"] = total_tokens
        logger.run.summary["num_fragments"] = len(fragments)
        logger.run.summary["num_shards"] = len(shards)
        logger.run.summary["ran_out"] = total_tokens < arguments.token_budget
        logger.finish()


# Required with spawn: each worker re-imports this file, and must not rerun main()
if __name__ == "__main__":
    main()
