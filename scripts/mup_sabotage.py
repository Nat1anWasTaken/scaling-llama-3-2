"""
Run scripts.mup_check with one piece of the muP / CompleteP implementation deliberately
broken, to confirm the check actually catches it. Nothing in src/ is edited: each break is
a monkeypatch applied in this process only.

  python -m scripts.mup_sabotage <break> <mup_check args...>

  readout    drop the 1/m_d logit multiplier          -> width: logits delta slope > 0
  init       constant-std hidden init (SP init)       -> width: branch rms slope ~ +0.5 at step 0
  muon-lr    Muon adjust_lr_fn = match_rms_adamw      -> width: spectral ratio grows with width
  branch     drop the 1/m_L residual-branch multiplier -> depth: branch delta slope ~ 0, not -1
  norm-fold  skip the final-norm fold in the export   -> export: assert_close fails

Not a break but a variant to try, using the same mechanism:

  zero-out   zero-init o_proj / down_proj, as in the CompleteP + Muon reference code
"""

import sys
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from scripts import mup_check
from src.model import LlamaCausalLM

original_build_model = mup_check.build_model
original_build_optimizer = mup_check.build_muon_adamw
original_export = LlamaCausalLM.export_plain_llama


def patch_model(edit: Callable[[LlamaCausalLM], None]) -> None:
    def build_model(*args: Any, **kwargs: Any) -> LlamaCausalLM:  # noqa: ANN401
        model = original_build_model(*args, **kwargs)
        edit(model)
        return model

    mup_check.build_model = build_model


def drop_readout(model: LlamaCausalLM) -> None:
    # forward reads self.width_multiplier at call time
    model.width_multiplier = 1.0


def drop_branch(model: LlamaCausalLM) -> None:
    # the o_proj / down_proj hooks read self.depth_multiplier at call time
    model.depth_multiplier = 1.0


@torch.no_grad()
def sp_init(model: LlamaCausalLM) -> None:
    for layer in model.model.layers:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=model.config.initializer_range)


@torch.no_grad()
def zero_out_projections(model: LlamaCausalLM) -> None:
    # every residual branch starts at exactly 0, so h_L = embedding at init
    for layer in model.model.layers:
        nn.init.zeros_(layer.self_attn.o_proj.weight)
        nn.init.zeros_(layer.mlp.down_proj.weight)


def break_muon_lr() -> None:
    def build_muon_adamw(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        optimizer = original_build_optimizer(*args, **kwargs)
        for group in optimizer.optimizers[0].param_groups:
            group["adjust_lr_fn"] = "match_rms_adamw"
        return optimizer

    mup_check.build_muon_adamw = build_muon_adamw


def break_norm_fold() -> None:
    @torch.no_grad()
    def export_plain_llama(self: LlamaCausalLM) -> Any:  # noqa: ANN401
        plain = original_export(self)
        # undo the final-norm fold, as if that line were missing
        plain.model.norm.weight.mul_(
            self.width_multiplier * self.config.embedding_multiplier
        )
        return plain

    LlamaCausalLM.export_plain_llama = export_plain_llama


BREAKS: dict[str, Callable[[], None]] = {
    "readout": lambda: patch_model(drop_readout),
    "init": lambda: patch_model(sp_init),
    "muon-lr": break_muon_lr,
    "branch": lambda: patch_model(drop_branch),
    "norm-fold": break_norm_fold,
    "zero-out": lambda: patch_model(zero_out_projections),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in BREAKS:
        sys.exit(f"usage: python -m scripts.mup_sabotage {{{','.join(BREAKS)}}} ...")
    name = sys.argv.pop(1)
    print(f"=== SABOTAGE: {name} ===")
    BREAKS[name]()
    mup_check.main()


if __name__ == "__main__":
    main()
