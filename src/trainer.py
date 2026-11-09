import contextlib
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel

from .data import Batch
from .loss import CausalLMLossSum


@dataclass
class TrainingConfig:
    total_tokens: int
    seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    max_grad_norm: float | None
    log_every_steps: int
    eval_every_steps: int
    ckpt_every_steps: int

    def __post_init__(self) -> None:
        positive_fields = (
            "total_tokens",
            "seq_len",
            "micro_batch_size",
            "grad_accum_steps",
            "log_every_steps",
            "eval_every_steps",
            "ckpt_every_steps",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0")
        if self.seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if self.total_tokens % self.seq_len != 0:
            raise ValueError("total_tokens must be divisible by seq_len")


@dataclass
class TrainerState:
    step: int = 0
    tokens_seen: int = 0


@dataclass
class EvaluationResult:
    loss: float
    perplexity: float


class Evaluator(Protocol):
    # The eval data and loss fn should be put into the evaluator when intialized
    def __call__(
        self,
        model: PreTrainedModel,
    ) -> EvaluationResult:
        raise NotImplementedError


class Checkpointer(Protocol):
    def save(self, trainer_state: TrainerState, model: PreTrainedModel) -> None:
        raise NotImplementedError


class Logger(Protocol):
    def log(self, data: dict[str, float], step: int) -> None:
        raise NotImplementedError


def autocast(device: torch.device) -> contextlib.AbstractContextManager[object]:
    """
    bf16 autocast on CUDA, plain fp32 elsewhere (CPU/MPS debug runs)
    """
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)

    return contextlib.nullcontext()


def uses_varlen_attention(model: PreTrainedModel) -> bool:
    """
    FlashAttention takes packed documents as one flat sequence plus their boundaries
    """
    return str(model.config._attn_implementation).startswith("flash_attention")


def forward_batch(
    model: torch.nn.Module, batch: Batch, device: torch.device, varlen: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the model on one packed batch, returning the logits and the labels they line up with
    """
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    position_ids = batch["position_ids"]
    varlen_kwargs: dict[str, torch.Tensor | int] = {}

    if varlen:
        # HF only infers document boundaries from position_ids when the batch size is 1,
        # and pays a GPU sync in every layer to do it. So flatten [B, T] into [1, B*T]
        # and hand FlashAttention the boundaries directly, computed here on the CPU.
        # Every row starts at position 0, so row edges count as document edges too
        seq_len = input_ids.shape[1]
        input_ids, labels, position_ids = (
            tensor.reshape(1, -1) for tensor in (input_ids, labels, position_ids)
        )
        starts = (position_ids[0] == 0).nonzero().flatten()
        cu_seq_lens = torch.cat([starts, torch.tensor([position_ids.numel()])]).to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        varlen_kwargs = {
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            # An upper bound is enough, and a constant avoids a recompile for every batch
            "max_length_q": seq_len,
            "max_length_k": seq_len,
        }

    # non_blocking: a blocking host-to-device copy first waits for every queued kernel
    with autocast(device):
        output = model(
            input_ids=input_ids.to(device, non_blocking=True),
            position_ids=position_ids.to(device, non_blocking=True),
            use_cache=False,
            **varlen_kwargs,
        )

    return output.logits, labels.to(device, non_blocking=True)


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place sum across ranks, unchanged on a single process
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)

    return tensor


def barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel | DistributedDataParallel,
        train_data: Iterable[Batch],
        loss_fn: CausalLMLossSum,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        evaluator: Evaluator | None,
        ckpt: Checkpointer,
        logger: Logger,
        cfg: TrainingConfig,
    ) -> None:
        # Note we do not perform any validation here, since we already do it in the config
        self.model = model
        self.raw_model = (
            cast(PreTrainedModel, model.module)
            if isinstance(model, DistributedDataParallel)
            else model
        )
        self.device = self.raw_model.device
        self.varlen = uses_varlen_attention(self.raw_model)
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        self.train_data = train_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.ckpt = ckpt
        self.logger = logger
        self.cfg = cfg
        self.state = TrainerState()

    def train(self) -> None:
        self.model.train()
        train_iter = iter(self.train_data)
        device = self.device

        # [loss_sum, loss_tokens, input_tokens] summed over all ranks since the last log.
        # float64: token counts reach 1e10, past where float32 holds integers exactly
        log_stats = torch.zeros(3, dtype=torch.float64, device=device)
        log_start_time = time.monotonic()

        while self.state.tokens_seen < self.cfg.total_tokens:
            self.optimizer.zero_grad()

            # Get remaining tokens and batch samples
            # tokens_seen counts all ranks, each rank only covers 1/world_size of it
            remaining_tokens = self.cfg.total_tokens - self.state.tokens_seen
            local_remaining_tokens = remaining_tokens // self.world_size
            batches = self._get_batch_samples(train_iter, local_remaining_tokens)
            if not batches:
                train_iter = iter(self.train_data)
                batches = self._get_batch_samples(train_iter, local_remaining_tokens)
                if not batches:
                    raise ValueError("No batches found even after resetting train_iter")

            # Same layout as log_stats, for this step only. Kept on the GPU so the
            # micro-batch loop never waits for the GPU to catch up
            step_stats = torch.zeros(3, dtype=torch.float64, device=device)

            # Compute loss and gradients for each batch
            for index, batch in enumerate(batches):
                is_last_micro_batch = index == len(batches) - 1

                sync_context = (
                    self.model.no_sync()
                    if isinstance(self.model, DistributedDataParallel)
                    and not is_last_micro_batch
                    else contextlib.nullcontext()
                )

                with sync_context:
                    logits, labels = forward_batch(
                        self.model, batch, device, self.varlen
                    )
                    loss_sum, num_loss_tokens = self.loss_fn(logits, labels)
                    loss_sum.backward()

                # Free the logits now, a for loop keeps them alive until the next forward returns
                del logits

                step_stats[0] += loss_sum.detach()
                step_stats[1] += num_loss_tokens
                step_stats[2] += batch["input_ids"].numel()

            # One all-reduce for all three numbers, so log_stats is already global
            all_reduce_sum(step_stats)
            global_loss_tokens = step_stats[1]

            # Normalize gradients by the loss tokens of all ranks
            # DDP already averaged them to sum_r(grad_r) / world_size, so multiply by
            # world_size / global_loss_tokens to get sum_r(grad_r) / global_loss_tokens.
            # scale is a 0-dim GPU tensor, so this doesn't need the value on the CPU
            scale = self.world_size / global_loss_tokens
            for param in self.raw_model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)

            # Clip gradients and update model
            # Every rank holds the same gradients after the all-reduce, so the norm matches too.
            # The norm stays a tensor and is only read back when logging
            max_norm = float("inf")
            if self.cfg.max_grad_norm is not None:
                max_norm = self.cfg.max_grad_norm
            grad_norm = clip_grad_norm_(self.raw_model.parameters(), max_norm=max_norm)

            # Next step
            self.optimizer.step()
            self.scheduler.step()
            log_stats += step_stats
            self.state.step += 1
            # The step's only GPU sync: the while condition needs tokens_seen on the CPU
            self.state.tokens_seen += int(step_stats[2].item())
            is_last_step = self.state.tokens_seen >= self.cfg.total_tokens

            # Log metrics
            # Note that we only log metrics at the end of each log_every_steps interval or when we reach total_tokens
            if self.state.step % self.cfg.log_every_steps == 0 or is_last_step:
                if self.rank == 0:
                    log_loss_sum, log_loss_tokens, log_input_tokens = log_stats.tolist()
                    elapsed = time.monotonic() - log_start_time
                    metrics = {
                        "train/loss": log_loss_sum / log_loss_tokens,
                        "train/grad_norm": grad_norm.item(),
                        "train/step": self.state.step,
                        "train/tokens_per_second": log_input_tokens / elapsed,
                        "train/total_tokens_seen": self.state.tokens_seen,
                    }
                    # One learning rate per param group, named when the group has a name
                    for index, group in enumerate(self.optimizer.param_groups):
                        name = group.get("name", str(index))
                        metrics[f"train/learning_rate/{name}"] = group["lr"]

                    self.logger.log(metrics, step=self.state.step)

                log_stats.zero_()
                log_start_time = time.monotonic()

            # Eval and checkpoint run on rank 0 with the unwrapped model, the other ranks
            # wait at the barrier. The DDP wrapper would block in its buffer broadcast
            if self.state.step % self.cfg.eval_every_steps == 0 or is_last_step:
                if self.rank == 0 and self.evaluator is not None:
                    result = self.evaluator(self.raw_model)
                    self.logger.log(
                        {
                            "eval/loss": result.loss,
                            "eval/perplexity": result.perplexity,
                        },
                        step=self.state.step,
                    )
                barrier()
                # Don't count eval time as training throughput
                log_start_time = time.monotonic()

            if self.state.step % self.cfg.ckpt_every_steps == 0 or is_last_step:
                if self.rank == 0:
                    self.ckpt.save(self.state, self.raw_model)
                barrier()
                log_start_time = time.monotonic()

    def _get_batch_samples(
        self,
        train_iter: Iterator[Batch],
        remaining_tokens: int,
    ) -> list[Batch]:
        """
        Samples batches from the train iterator until the remaining tokens equal or exceed 0.
        """
        batches: list[Batch] = []
        # Sample batches until we have enough tokens or run out of data
        for _ in range(self.cfg.grad_accum_steps):
            if remaining_tokens <= 0:
                break
            batch = next(train_iter, None)
            if batch is None:
                break
            batch_size = batch["input_ids"].shape[0]
            n_sequences = min(batch_size, remaining_tokens // self.cfg.seq_len)

            if n_sequences < batch_size:
                trimmed_batch: Batch = {
                    "input_ids": batch["input_ids"][:n_sequences],
                    "labels": batch["labels"][:n_sequences],
                    "position_ids": batch["position_ids"][:n_sequences],
                }
                batch = trimmed_batch
            batches.append(batch)
            remaining_tokens -= batch["input_ids"].numel()

        return batches

    def state_dict(self) -> TrainerState:
        raise NotImplementedError

    def load_state_dict(self, state_dict: TrainerState) -> None:
        raise NotImplementedError
