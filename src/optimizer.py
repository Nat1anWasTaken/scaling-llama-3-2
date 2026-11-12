import math
from collections.abc import Callable
from typing import Any, overload

from torch import nn
from torch.optim import AdamW, Muon, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.model import LlamaCausalLM


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

        # Optimizer.load_state_dict swaps in new group dicts. Point at them again,
        # or the scheduler keeps updating the old dicts and the lr stops changing
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]


def build_muon_adamw(
    model: LlamaCausalLM,
    muon_lr: float,
    adamw_lr: float,
    muon_weight_decay: float,
    adamw_weight_decay: float,
    adamw_betas: tuple[float, float],
    adamw_eps: float = 1e-12,
) -> CombinedOptimizer:
    """
    Muon in spectral units on every 2D weight inside the decoder layers, AdamW on the
    tied embedding and the RMSNorm gains. Width-independent lrs: the per-matrix
    sqrt(fan_out / fan_in) factor is the muP rule for Muon, the muP readout multiplier
    lives in the model. Only the hidden weight decay is width-scaled, by 1 / m_in, so the
    per-step decay lr * wd of each hidden matrix shrinks with its fan_in
    """
    config = model.config
    if config.hidden_size == config.intermediate_size:
        raise ValueError("hidden_size == intermediate_size makes fan_in ambiguous")

    # fan_in -> width multiplier m_in, mirroring LlamaCausalLM._init_weights
    width_multipliers = {
        config.hidden_size: config.hidden_size / config.base_hidden_size,
        config.intermediate_size: config.intermediate_size
        / config.base_intermediate_size,
    }
    muon_parameters: dict[int, list[nn.Parameter]] = {k: [] for k in width_multipliers}
    adamw_decay_parameters = []
    adamw_no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if parameter.ndim == 2 and ".layers." in f".{name}":
            fan_in = parameter.shape[1]
            if fan_in not in muon_parameters:
                raise ValueError(f"{name}: unexpected fan_in {fan_in}")
            muon_parameters[fan_in].append(parameter)
        elif parameter.ndim == 2:
            # the tied embedding / output layer
            adamw_decay_parameters.append(parameter)
        else:
            adamw_no_decay_parameters.append(parameter)

    muon = Muon(
        [
            {
                "params": parameters,
                "weight_decay": muon_weight_decay / width_multipliers[fan_in],
                "name": f"muon_fan_in_{fan_in}",
            }
            for fan_in, parameters in muon_parameters.items()
        ],
        lr=muon_lr,
        adjust_lr_fn="spectral_unclamped",
    )
    adamw = AdamW(
        [
            {
                "params": adamw_decay_parameters,
                "weight_decay": adamw_weight_decay,
                "name": "adamw",
            },
            {
                "params": adamw_no_decay_parameters,
                "weight_decay": 0.0,
                "name": "adamw_no_decay",
            },
        ],
        lr=adamw_lr,
        betas=adamw_betas,
        eps=adamw_eps,
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
