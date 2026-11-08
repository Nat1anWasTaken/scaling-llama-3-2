import math
from collections.abc import Callable
from typing import Any, overload

from torch import nn
from torch.optim import AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CombinedOptimizer(Optimizer):
    def __init__(self, optimizers: list[Optimizer]) -> None:
        self.optimizers = optimizers
        param_groups = [
            group for optimizer in optimizers for group in optimizer.param_groups
        ]
        super().__init__(param_groups, defaults={})

    # Same overloads as Optimizer.step. No torch.no_grad() needed here: each inner
    # optimizer's step already runs without grad
    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            raise ValueError("CombinedOptimizer does not support closures")

        for optimizer in self.optimizers:
            optimizer.step()

        return None

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer, inner_state in zip(
            self.optimizers, state_dict["optimizers"], strict=True
        ):
            optimizer.load_state_dict(inner_state)


def build_muon_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    adamw_betas: tuple[float, float],
) -> CombinedOptimizer:
    muon_parameters = []
    adamw_decay_parameters = []
    adamw_no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim == 2 and ".layers." in f".{name}":
            muon_parameters.append(parameter)
        elif parameter.ndim == 2:
            # the embedding & the output layer
            adamw_decay_parameters.append(parameter)
        else:
            adamw_no_decay_parameters.append(parameter)

    muon = Muon(
        [{"params": muon_parameters, "name": "muon"}],
        lr=learning_rate,
        weight_decay=weight_decay,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = AdamW(
        [
            {
                "params": adamw_decay_parameters,
                "weight_decay": weight_decay,
                "name": "adamw",
            },
            {
                "params": adamw_no_decay_parameters,
                "weight_decay": 0.0,
                "name": "adamw_no_decay",
            },
        ],
        lr=learning_rate,
        betas=adamw_betas,
    )

    return CombinedOptimizer([muon, adamw])


def build_warmup_stable_decay(
    optimizer: Optimizer, total_steps: int, warmup_steps: int, decay_fraction: float
) -> LambdaLR:
    decay_steps = max(1, math.ceil(total_steps * decay_fraction))
    decay_start = total_steps - decay_steps

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step < decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / decay_steps)

    return LambdaLR(optimizer, multiplier)
