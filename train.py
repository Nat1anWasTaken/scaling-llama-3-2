import argparse
import dataclasses
import math
import os
from collections import namedtuple
from pathlib import Path

import torch

from src.checkpoint import PretrainedCheckpointer
from src.data import PackedDataLoader, SequencePacker
from src.evaluation import DevelopmentEvaluator, load_development_texts
from src.loggers import ConsoleLogger, WandbLogger
from src.loss import CrossEntropyLossSum
from src.model import LlamaCausalLM, ScalingLlamaConfig
from src.optimizer import build_muon_adamw, build_warmup_stable_decay
from src.token_stream import TokenStream
from src.tokenizer import ShardStore, load_tokenizer
from src.trainer import Trainer, TrainingConfig, barrier

TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
DEVELOPMENT_DATASET = "cerulean-works/fineweb-edu-dev"
WANDB_PROJECT = "lab6-predictable-scaling"
VOCAB_SIZE = 128256


DistributedData = namedtuple("DistributedData", ["rank", "world_size", "device"])


def setup_distributed(requested: str | None) -> DistributedData:
    if "WORLD_SIZE" not in os.environ:
        return DistributedData(rank=0, world_size=1, device=pick_device(requested))

    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available() and requested != "cpu":
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        torch.distributed.init_process_group("gloo")
        device = torch.device("cpu")

    return DistributedData(
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        device=device,
    )


@dataclasses.dataclass(frozen=True)
class ModelSize:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_query_heads: int
    num_key_value_heads: int


# The reference configurations from LAB.md, plus a tiny one for local smoke tests
MODEL_SIZES = {
    "debug": ModelSize(2, 128, 512, 4, 2),
    "150M": ModelSize(6, 768, 3072, 12, 3),
    "300M": ModelSize(9, 1024, 5184, 16, 4),
    "500M": ModelSize(11, 1536, 4672, 24, 6),
    "750M": ModelSize(13, 1792, 5888, 28, 7),
    "1B": ModelSize(15, 1792, 8000, 28, 7),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--data", required=True, help="shard store from prepare.py")
    parser.add_argument("--model-size", choices=MODEL_SIZES, required=True)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    # Replaces --grad-accum-steps: the global batch is part of the recipe, while the
    # micro batch and GPU count only decide how it is computed
    parser.add_argument(
        "--global-batch-tokens",
        type=int,
        default=4096 * 8 * 16 * 8,
        help="tokens per optimizer step summed over all ranks (default: 4,194,304)",
    )

    # muP: hyperparameters are tuned on --mup-base-size and transferred. Muon's lr is in
    # spectral units (update spectral norm per step), not comparable to the AdamW lr
    parser.add_argument(
        "--mup-base-size",
        choices=MODEL_SIZES,
        help="model size whose hyperparameters this run transfers from (default: none, "
        "i.e. plain SP with every multiplier at 1)",
    )
    parser.add_argument("--embedding-multiplier", type=float, default=1.0)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adamw-lr", type=float, default=3e-3)
    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=0.015,
        help="per-step hidden decay is muon_lr * this, scaled by 1 / m_in per matrix",
    )
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--adamw-beta1", type=float, default=0.9)
    parser.add_argument("--adamw-beta2", type=float, default=0.95)
    parser.add_argument("--adamw-eps", type=float, default=1e-12)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--decay-fraction", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--eval-every-steps", type=int, help="default: every 10%% of training"
    )
    parser.add_argument("--eval-max-batches", type=int, help="default: whole dev set")
    parser.add_argument(
        "--ckpt-every-steps", type=int, help="default: about every 5%% of training"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the newest checkpoint in the run's checkpoint directory",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--run-name",
        required=True,
        help="names the wandb run and the checkpoint directory, so runs never overwrite",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--development-dataset", default=DEVELOPMENT_DATASET)
    parser.add_argument("--device", help="default: cuda, else mps, else cpu")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the model, on by default (--no-compile turns it off)",
    )
    parser.add_argument(
        "--attn-implementation",
        help="default: flash_attention_3 on CUDA, else sdpa",
    )
    parser.add_argument("--no-wandb", action="store_true", help="log to console only")

    return parser.parse_args()


def default_ckpt_every_steps(total_steps: int, log_every_steps: int) -> int:
    """
    About every 5% of training, rounded down to a multiple of the log interval
    """
    every = total_steps // 20 // log_every_steps * log_every_steps
    return max(every, log_every_steps)


def pick_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_config(
    size: ModelSize,
    base_size: ModelSize | None,
    embedding_multiplier: float,
    seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
) -> ScalingLlamaConfig:
    base_size = base_size or size
    return ScalingLlamaConfig(
        base_hidden_size=base_size.hidden_size,
        base_intermediate_size=base_size.intermediate_size,
        base_num_hidden_layers=base_size.num_layers,
        embedding_multiplier=embedding_multiplier,
        vocab_size=VOCAB_SIZE,
        hidden_size=size.hidden_size,
        intermediate_size=size.intermediate_size,
        num_hidden_layers=size.num_layers,
        num_attention_heads=size.num_query_heads,
        num_key_value_heads=size.num_key_value_heads,
        max_position_embeddings=seq_len,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        rms_norm_eps=1e-5,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)

    rank, world_size, device = setup_distributed(arguments.device)

    tokens_per_micro_step = arguments.seq_len * arguments.micro_batch_size * world_size
    if arguments.global_batch_tokens % tokens_per_micro_step != 0:
        raise ValueError(
            f"global_batch_tokens ({arguments.global_batch_tokens}) must be divisible by "
            f"seq_len * micro_batch_size * world_size ({tokens_per_micro_step})"
        )

    grad_accum_steps = arguments.global_batch_tokens // tokens_per_micro_step
    tokens_per_step = arguments.global_batch_tokens

    if arguments.total_tokens % tokens_per_step != 0:
        raise ValueError(
            f"total_tokens ({arguments.total_tokens}) must be divisible by "
            f"tokens_per_step ({tokens_per_step})"
        )

    total_steps = math.ceil(arguments.total_tokens / tokens_per_step)

    training_config = TrainingConfig(
        total_tokens=arguments.total_tokens,
        seq_len=arguments.seq_len,
        micro_batch_size=arguments.micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=arguments.max_grad_norm,
        log_every_steps=arguments.log_every_steps,
        eval_every_steps=arguments.eval_every_steps or max(1, total_steps // 10),
        ckpt_every_steps=arguments.ckpt_every_steps
        or default_ckpt_every_steps(total_steps, arguments.log_every_steps),
    )

    tokenizer = load_tokenizer(arguments.tokenizer)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must have bos_token_id and eos_token_id")

    model_config = build_model_config(
        MODEL_SIZES[arguments.model_size],
        MODEL_SIZES[arguments.mup_base_size] if arguments.mup_base_size else None,
        arguments.embedding_multiplier,
        arguments.seq_len,
        bos_token_id,
        eos_token_id,
    )
    model = LlamaCausalLM(model_config).to(device)

    # FA3 needs Hopper-class CUDA GPUs, so CPU and MPS debug runs fall back to SDPA
    attn_implementation = arguments.attn_implementation or (
        "flash_attention_3" if device.type == "cuda" else "sdpa"
    )
    model.set_attn_implementation(attn_implementation)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    if rank == 0:
        print(
            f"model: {arguments.model_size}, {num_parameters:,} parameters on {device}"
        )

    optimizer = build_muon_adamw(
        model,
        muon_lr=arguments.muon_lr,
        adamw_lr=arguments.adamw_lr,
        muon_weight_decay=arguments.muon_weight_decay,
        adamw_weight_decay=arguments.adamw_weight_decay,
        adamw_betas=(arguments.adamw_beta1, arguments.adamw_beta2),
        adamw_eps=arguments.adamw_eps,
    )
    scheduler = build_warmup_stable_decay(
        optimizer,
        total_steps=total_steps,
        warmup_steps=arguments.warmup_steps,
        decay_fraction=arguments.decay_fraction,
    )

    training_model = model

    if world_size > 1:
        device_ids = [device.index] if device.type == "cuda" else None
        training_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=device_ids, output_device=device
        )

    # Compile after wrapping, so dynamo sees DDP and keeps all-reduce overlapped with backward
    if arguments.compile:
        training_model.compile()
        # eval runs on the unwrapped model, which the DDP compile above
        # doesn't cover. Compile it too, so eval uses the same kernels as on 1 GPU
        if training_model is not model:
            model.compile()

    loss_fn = CrossEntropyLossSum()
    # The stream shards documents across ranks before packing, so each rank only
    # packs its own 1/world_size
    documents = TokenStream(
        store=ShardStore(arguments.data),
        seed=arguments.seed,
        rank=rank,
        world_size=world_size,
    )
    rows = SequencePacker(documents, arguments.seq_len)
    train_data = PackedDataLoader(rows, arguments.micro_batch_size)

    # Every rank evaluates its share of the dev set, so every rank needs an evaluator:
    # a rank without one would skip the eval all-reduce and hang the others.
    # Rank 0 downloads the dev set first, so the other ranks read it from the HF cache
    if rank == 0:
        load_development_texts(arguments.development_dataset)
    barrier()
    evaluator = DevelopmentEvaluator(
        tokenizer,
        arguments.development_dataset,
        loss_fn,
        seq_len=arguments.seq_len,
        micro_batch_size=arguments.micro_batch_size,
        max_batches=arguments.eval_max_batches,
        rank=rank,
        world_size=world_size,
    )

    run_config: dict[str, object] = {
        **vars(arguments),
        "output_dir": str(arguments.output_dir),
        "model": model_config.to_dict(),
        "num_parameters": num_parameters,
        "tokens_per_step": tokens_per_step,
        "world_size": world_size,
        "grad_accum_steps": grad_accum_steps,
        "total_steps": total_steps,
        "training": dataclasses.asdict(training_config),
        "initialization": (
            f"muP: hidden normal(0, {model_config.initializer_range} * sqrt(base_fan_in "
            "/ fan_in)), embedding constant std; logits * base_d / d, residual branches "
            "* base_L / L (CompleteP alpha=1), embedding * embedding_multiplier"
        ),
        "optimizer": (
            "Muon(spectral_unclamped) for decoder 2D weights with wd / m_in, "
            "AdamW for the tied embedding and RMSNorm gains"
        ),
        "schedule": "linear warmup, constant, linear decay to 0 (WSD)",
        "attention": attn_implementation,
    }

    run_dir = arguments.output_dir / arguments.run_name
    checkpointer = PretrainedCheckpointer(run_dir, tokenizer)
    resume_from = checkpointer.latest()

    # Checked before wandb starts, so a refused launch leaves no empty run behind
    if arguments.resume and resume_from is None:
        raise FileNotFoundError(f"--resume, but {run_dir} holds no checkpoint")
    if not arguments.resume and resume_from is not None:
        raise FileExistsError(
            f"{run_dir} already holds {resume_from.name}; pass --resume to continue "
            "it, or pick a new --run-name"
        )

    if arguments.no_wandb or rank != 0:
        logger: ConsoleLogger | WandbLogger = ConsoleLogger()
    else:
        logger = WandbLogger(WANDB_PROJECT, arguments.run_name, run_config)

    trainer = Trainer(
        model=training_model,
        train_data=train_data,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        ckpt=checkpointer,
        logger=logger,
        cfg=training_config,
    )

    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
        if rank == 0:
            print(f"resumed from {resume_from}")

    if rank == 0:
        print(f"training {total_steps} steps of {tokens_per_step:,} tokens")
    trainer.train()

    if isinstance(logger, WandbLogger):
        logger.finish()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
