import contextlib
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch
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


class Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        train_data: Iterable[Batch],
        loss_fn: CausalLMLossSum,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        evaluator: Evaluator,
        ckpt: Checkpointer,
        logger: Logger,
        cfg: TrainingConfig,
    ) -> None:
        # Note we do not perform any validation here, since we already do it in the config
        self.model = model
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
        device = self.model.device

        log_loss_sum = 0
        log_loss_token = 0
        log_input_tokens = 0
        log_start_time = time.monotonic()

        while self.state.tokens_seen < self.cfg.total_tokens:
            self.optimizer.zero_grad()

            # Get remaining tokens and batch samples
            remaining_tokens = self.cfg.total_tokens - self.state.tokens_seen
            batches = self._get_batch_samples(train_iter, remaining_tokens)
            if not batches:
                train_iter = iter(self.train_data)
                batches = self._get_batch_samples(train_iter, remaining_tokens)
                if not batches:
                    raise ValueError("No batches found even after resetting train_iter")

            # Create log metrics
            update_loss_tokens = 0
            update_input_tokens = 0

            # Compute loss and gradients for each batch
            for batch in batches:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                position_ids = batch["position_ids"].to(device)

                with autocast(device):
                    output = self.model(
                        input_ids=input_ids, position_ids=position_ids, use_cache=False
                    )
                logits = output.logits
                loss_sum, num_loss_tokens = self.loss_fn(logits, labels)
                loss_sum.backward()

                log_loss_sum += loss_sum.detach().item()
                update_loss_tokens += num_loss_tokens.item()
                update_input_tokens += input_ids.numel()

            # Normalize gradients and update model
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.div_(update_loss_tokens)

            # Clip gradients and update model
            max_norm = float("inf")
            if self.cfg.max_grad_norm is not None:
                max_norm = self.cfg.max_grad_norm
            grad_norm: float = clip_grad_norm_(
                self.model.parameters(), max_norm=max_norm
            ).item()

            # Next step
            self.optimizer.step()
            self.scheduler.step()
            self.state.step += 1
            self.state.tokens_seen += update_input_tokens
            log_loss_token += update_loss_tokens
            log_input_tokens += update_input_tokens
            is_last_step = self.state.tokens_seen >= self.cfg.total_tokens

            # Log metrics
            # Note that we only log metrics at the end of each log_every_steps interval or when we reach total_tokens
            if self.state.step % self.cfg.log_every_steps == 0 or is_last_step:
                elapsed = time.monotonic() - log_start_time
                metrics = {
                    "train/loss": log_loss_sum / log_loss_token,
                    "train/grad_norm": grad_norm,
                    "train/step": self.state.step,
                    "train/tokens_per_second": log_input_tokens / elapsed,
                    "train/total_tokens_seen": self.state.tokens_seen,
                }
                # One learning rate per param group, named when the group has a name
                for index, group in enumerate(self.optimizer.param_groups):
                    name = group.get("name", str(index))
                    metrics[f"train/learning_rate/{name}"] = group["lr"]

                self.logger.log(metrics, step=self.state.step)
                log_loss_sum = 0
                log_loss_token = 0
                log_input_tokens = 0
                log_start_time = time.monotonic()

            if self.state.step % self.cfg.eval_every_steps == 0 or is_last_step:
                result = self.evaluator(self.model)
                self.logger.log(
                    {"eval/loss": result.loss, "eval/perplexity": result.perplexity},
                    step=self.state.step,
                )
                # Don't count eval time as training throughput
                log_start_time = time.monotonic()

            if self.state.step % self.cfg.ckpt_every_steps == 0 or is_last_step:
                self.ckpt.save(self.state, self.model)
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
