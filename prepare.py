"""
Pre-tokenize FineWeb-Edu into shards, in a fixed shuffled fragment order, until a token budget.

    eval "$(flock s3 credentials --ttl 24h)"
    uv run prepare.py --output s3://<bucket>/fineweb-edu-llama3 --token-budget 30000000000

Rerunning with the same --output skips finished shards, so an interrupted run (or an
expired S3 key) only needs a rerun, and a larger budget only tokenizes the extra fragments.
"""

import argparse
import multiprocessing
import os
import time

from src.tokenizer import (
    FragmentTokenizer,
    ShardStore,
    load_development_ids,
    load_tokenizer,
    open_dataset,
    shuffled_fragment_ids,
)

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
DATASET_URI = "s3://flock-datasets/fineweb-edu"
DEVELOPMENT_DATASET = "cerulean-works/fineweb-edu-dev"
DEVELOPMENT_SIZE = 5000
SHARD_FORMAT = "[BOS, text, EOS] int32 tokens.npy + int64 offsets.npy"

# One per worker process, built by initialize_worker
worker: FragmentTokenizer | None = None


def initialize_worker(
    dataset_uri: str,
    tokenizer_name: str,
    exclude_ids: frozenset[str],
    output_uri: str,
    batch_size: int,
) -> None:
    global worker

    # Each process already uses one core; tokenizer threads on top would oversubscribe
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    worker = FragmentTokenizer(
        open_dataset(dataset_uri),
        load_tokenizer(tokenizer_name),
        exclude_ids,
        ShardStore(output_uri),
        batch_size,
    )


def tokenize_fragment(fragment_id: int) -> tuple[int, int]:
    if worker is None:
        raise RuntimeError("worker is not initialized")

    return fragment_id, worker.tokenize_fragment(fragment_id)


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--dataset", default=DATASET_URI)
    parser.add_argument("--development-dataset", default=DEVELOPMENT_DATASET)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    store = ShardStore(arguments.output)
    dataset = open_dataset(arguments.dataset)

    check_metadata(
        store,
        {
            "tokenizer": arguments.tokenizer,
            "dataset": arguments.dataset,
            "dataset_version": dataset.version,
            "development_dataset": arguments.development_dataset,
            "format": SHARD_FORMAT,
        },
    )

    # Download into the HF cache once here, so workers don't all race to download it,
    load_tokenizer(arguments.tokenizer)

    exclude_ids = load_development_ids(arguments.development_dataset, DEVELOPMENT_SIZE)
    fragment_ids = shuffled_fragment_ids(dataset, arguments.seed)

    shards: list[dict[str, int]] = []
    total_tokens = 0
    start_time = time.monotonic()

    # spawn, not fork: this process already runs Lance's background threads, and
    # forking a process with live threads can deadlock the child
    context = multiprocessing.get_context("spawn")

    with context.Pool(
        arguments.workers,
        initializer=initialize_worker,
        initargs=(
            arguments.dataset,
            arguments.tokenizer,
            exclude_ids,
            store.uri,
            arguments.batch_size,
        ),
    ) as pool:
        # imap yields in fragment_ids order however the workers finish, so the
        # selected shards are always the same prefix of the shuffled order
        for fragment_id, tokens in pool.imap(tokenize_fragment, fragment_ids):
            shards.append({"fragment_id": fragment_id, "tokens": tokens})
            total_tokens += tokens

            elapsed = time.monotonic() - start_time
            print(
                f"[{len(shards)}] fragment {fragment_id:06d}: {tokens / 1e6:.1f}M tokens, "
                f"total {total_tokens / 1e9:.2f}B / {arguments.token_budget / 1e9:.2f}B "
                f"({total_tokens / elapsed / 1e6:.2f}M tokens/s)",
                flush=True,
            )

            if total_tokens >= arguments.token_budget:
                break

        # Leaving the with block terminates workers still busy on later fragments.
        # Their shards never got offsets.npy, so a rerun redoes them

    if total_tokens < arguments.token_budget:
        print(f"warning: dataset ran out at {total_tokens} tokens", flush=True)

    store.write_json(
        "manifest.json",
        {
            "seed": arguments.seed,
            "token_budget": arguments.token_budget,
            "total_tokens": total_tokens,
            "shards": shards,
        },
    )


# Required with spawn: each worker re-imports this file, and must not rerun main()
if __name__ == "__main__":
    main()
