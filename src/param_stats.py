import re
from collections import defaultdict

import torch

_LAYER_INDEX = re.compile(r"layers\.(\d+)\.")


def stat_keys(name: str) -> tuple[str, str]:
    """
    model.layers.3.self_attn.q_proj.weight -> ("L03/q_proj", "all/q_proj")
    """
    kind = "embed" if "embed_tokens" in name else name.split(".")[-2]
    match = _LAYER_INDEX.search(name)
    layer_key = f"L{int(match.group(1)):02d}/{kind}" if match else kind
    return layer_key, f"all/{kind}"


class ParameterStats:
    """
    Weight, update and gradient norms of every trainable parameter, summed per layer and
    per matrix kind over all layers.

    The GPU side only produces one [3, num_params] tensor of per-parameter norms; the
    caller reads it back together with its other log values in one sync, and the grouping
    happens on the CPU. Runs eagerly outside the compiled model, so it never recompiles.

    Assumes DDP: every rank holds the full, identical gradients after the all-reduce, so
    one rank's numbers are the global ones. Under FSDP each rank would only see its shard
    """

    def __init__(self, model: torch.nn.Module) -> None:
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in named]
        self.keys = [stat_keys(name) for name, _ in named]

        self.numel: dict[str, int] = defaultdict(int)
        for keys, param in zip(self.keys, self.params):
            for key in keys:
                self.numel[key] += param.numel()

    @torch.no_grad()
    def snapshot(self) -> list[torch.Tensor]:
        # Same dtype as the parameters (fp32): a bf16 copy would round away updates
        # smaller than ~0.4% of the weight, which is most of them
        return [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def norms(self, before: list[torch.Tensor], grad_norms: torch.Tensor) -> torch.Tensor:
        """
        [3, num_params]: weight, update and (pre-clip) gradient norms. Call after
        optimizer.step(); overwrites before with (before - after) to avoid another copy
        """
        torch._foreach_sub_(before, self.params)
        weight_norms = torch._foreach_norm(self.params, dtype=torch.float32)
        update_norms = torch._foreach_norm(before, dtype=torch.float32)
        return torch.stack([torch.stack(weight_norms), torch.stack(update_norms), grad_norms])

    def to_metrics(self, values: list[float]) -> dict[str, float]:
        """
        values is norms()'s result, flattened and read back to the CPU
        """
        num_params = len(self.params)
        weight_sq: dict[str, float] = defaultdict(float)
        update_sq: dict[str, float] = defaultdict(float)
        grad_sq: dict[str, float] = defaultdict(float)

        for i, keys in enumerate(self.keys):
            weight = values[i]
            update = values[num_params + i]
            grad = values[2 * num_params + i]
            for key in keys:
                weight_sq[key] += weight * weight
                update_sq[key] += update * update
                grad_sq[key] += grad * grad

        metrics = {}
        for key, numel in self.numel.items():
            metrics[f"params/rms/{key}"] = (weight_sq[key] / numel) ** 0.5
            metrics[f"params/update_rms/{key}"] = (update_sq[key] / numel) ** 0.5
            metrics[f"params/update_ratio/{key}"] = (
                update_sq[key] / max(weight_sq[key], 1e-30)
            ) ** 0.5
            metrics[f"params/grad_norm/{key}"] = grad_sq[key] ** 0.5
        return metrics
