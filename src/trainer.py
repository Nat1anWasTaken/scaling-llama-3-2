from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from typing import Protocol

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


@dataclass
class TrainerState:
    step: int
    trainer: InitVar["Trainer | None"] = None

    @property
    def tokens_seen(self) -> int:
        # Derive from trainer's state, micro_batch_size * seq_len * grad_accum_steps * world_size * step
        raise NotImplementedError

    # TODO: extend this to include optimizer, scheduler, rng state.


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
        raise NotImplementedError()

    def train(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> TrainerState:
        raise NotImplementedError

    def load_state_dict(self, state_dict: TrainerState) -> None:
        raise NotImplementedError
