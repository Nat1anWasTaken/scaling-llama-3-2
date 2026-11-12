"""
Tiny-model checks that our muP / CompleteP implementation behaves as intended:

  export  export_plain_llama() and torch.compile reproduce our forward's logits
  width   coordinate check over width at fixed depth (muP)
  depth   coordinate check over depth at fixed width (CompleteP)
  joint   coordinate check over the real MODEL_SIZES ladder, width and depth together,
          against the first size as base (x axis: non-embedding params, in millions)

Add --no-mup to width/depth for the control run with every multiplier at 1.
"""

import argparse
import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.optim import Optimizer

from src.model import LlamaCausalLM
from src.optimizer import build_muon_adamw
from train import MODEL_SIZES, VOCAB_SIZE, ModelSize, build_model_config

HEAD_DIM = 64
DEVICE = torch.device("cuda")

Stats = dict[str, tuple[float, float]]


def make_size(hidden: int, layers: int) -> ModelSize:
    # head_dim fixed at 64 like MODEL_SIZES, so width grows by adding heads
    heads = hidden // HEAD_DIM
    return ModelSize(layers, hidden, 4 * hidden, heads, max(1, heads // 4))


def build_model(
    size: ModelSize, base: ModelSize | None, alpha: float, seq_len: int
) -> LlamaCausalLM:
    config = build_model_config(size, base, alpha, seq_len, 0, 1)
    model = LlamaCausalLM(config).to(DEVICE)
    model.set_attn_implementation("sdpa")
    return model


class Probes:
    """Forward hooks that keep a float copy of each tracked activation, only while enabled"""

    def __init__(self, model: LlamaCausalLM) -> None:
        self.enabled = False
        self.acts: dict[str, torch.Tensor] = {}
        self._watch(model.model.embed_tokens, "embed")
        for i, layer in enumerate(model.model.layers):
            # registered after LlamaCausalLM's own hooks, so these see the 1/m_L-scaled output
            self._watch(layer.self_attn.o_proj, f"attn_out.{i}")
            self._watch(layer.mlp.down_proj, f"mlp_out.{i}")
            self._watch(layer, f"resid.{i}")
        self._watch(model.model.norm, "final_norm")

    def _watch(self, module: nn.Module, name: str) -> None:
        def hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:  # noqa: ANN401
            if self.enabled:
                self.acts[name] = output.detach().float()

        module.register_forward_hook(hook)


@torch.no_grad()
def measure(
    model: LlamaCausalLM, probes: Probes, tokens: torch.Tensor
) -> dict[str, torch.Tensor]:
    model.eval()
    probes.enabled = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=tokens).logits
    probes.enabled = False
    model.train()
    return {**probes.acts, "logits": logits.float()}


def summarize(
    acts: dict[str, torch.Tensor], init: dict[str, torch.Tensor], num_layers: int
) -> Stats:
    """(rms of the activation, rms of its change since init) per tracked coordinate"""

    def stat(name: str) -> tuple[float, float]:
        x, x0 = acts[name], init[name]
        return x.pow(2).mean().sqrt().item(), (x - x0).pow(2).mean().sqrt().item()

    def layer_mean(kind: str) -> tuple[float, float]:
        stats = [stat(f"{kind}.{i}") for i in range(num_layers)]
        return (
            sum(s[0] for s in stats) / num_layers,
            sum(s[1] for s in stats) / num_layers,
        )

    return {
        "embed": stat("embed"),
        "attn_out": layer_mean("attn_out"),
        "mlp_out": layer_mean("mlp_out"),
        "resid_last": stat(f"resid.{num_layers - 1}"),
        "final_norm": stat("final_norm"),
        "logits": stat("logits"),
    }


@torch.no_grad()
def spectral_ratios(
    muon: Optimizer, before: dict[torch.Tensor, torch.Tensor]
) -> list[float]:
    """
    ||ΔW||_2 / (lr * sqrt(fan_out / fan_in)) per Muon matrix, with the decoupled weight
    decay removed first. The spectral condition says this is O(1) at every width
    """
    ratios = []
    for group in muon.param_groups:
        lr, wd = group["lr"], group["weight_decay"]
        for p in group["params"]:
            fan_out, fan_in = p.shape
            update = p.float() - before[p] * (1 - lr * wd)
            sigma = torch.linalg.matrix_norm(update, ord=2).item()
            ratios.append(sigma / (lr * math.sqrt(fan_out / fan_in)))
    return ratios


def zipf_tokens(
    generator: torch.Generator, batch_size: int, length: int
) -> torch.Tensor:
    # skewed like real text so there's a unigram signal to learn; real shards work too
    probs = 1.0 / torch.arange(1, VOCAB_SIZE + 1, dtype=torch.float)
    ids = torch.multinomial(
        probs, batch_size * length, replacement=True, generator=generator
    )
    return ids.view(batch_size, length).to(DEVICE)


def run_one(
    size: ModelSize, base: ModelSize | None, args: argparse.Namespace
) -> dict[str, Any]:
    histories: list[list[Stats]] = []
    ratios: list[float] = []
    losses: list[float] = []
    for seed in range(args.seeds):
        # seeds vary the init only; every config sees the same batches
        torch.manual_seed(seed)
        model = build_model(size, base, args.embedding_multiplier, args.seq_len)
        probes = Probes(model)
        optimizer = build_muon_adamw(
            model,
            muon_lr=args.muon_lr,
            adamw_lr=args.adamw_lr,
            muon_weight_decay=args.muon_weight_decay,
            adamw_weight_decay=args.adamw_weight_decay,
            adamw_betas=(0.9, 0.95),
        )
        muon = optimizer.optimizers[0]

        data = torch.Generator().manual_seed(1234)
        eval_tokens = zipf_tokens(data, args.batch_size, args.seq_len)
        init = measure(model, probes, eval_tokens)
        history = [summarize(init, init, size.num_layers)]

        loss = torch.tensor(float("nan"))
        for _ in range(args.steps):
            tokens = zipf_tokens(data, args.batch_size, args.seq_len + 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=tokens[:, :-1]).logits
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), tokens[:, 1:].flatten()
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            before = {
                p: p.detach().float().clone()
                for g in muon.param_groups
                for p in g["params"]
            }
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ratios += spectral_ratios(muon, before)
            history.append(
                summarize(measure(model, probes, eval_tokens), init, size.num_layers)
            )
            del before

        histories.append(history)
        losses.append(loss.item())
        del model, optimizer, muon, probes, init
        torch.cuda.empty_cache()

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    averaged = [
        {
            m: (
                mean([h[t][m][0] for h in histories]),
                mean([h[t][m][1] for h in histories]),
            )
            for m in histories[0][t]
        }
        for t in range(args.steps + 1)
    ]
    return {
        "history": averaged,
        "ratio_mean": mean(ratios),
        "ratio_max": max(ratios),
        "loss": mean(losses),
    }


def non_embedding_millions(size: ModelSize) -> int:
    hidden, inter = size.hidden_size, size.intermediate_size
    attention = 2 * hidden * hidden + 2 * hidden * size.num_key_value_heads * HEAD_DIM
    return round(size.num_layers * (attention + 3 * hidden * inter) / 1e6)


def slope(xs: list[int], ys: list[float]) -> float:
    """least-squares slope of log y on log x: 0 means the coordinate doesn't scale"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(max(y, 1e-30)) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly, strict=True)) / sum(
        (a - mx) ** 2 for a in lx
    )


def report(
    label: str, xs: list[int], results: list[dict[str, Any]], steps: int
) -> None:
    print(f"{label}: {xs}")
    shown = {0, 1, steps // 2, steps} | {t for t in (5, 10) if t <= steps}
    for t in sorted(shown):
        print(f"\nstep {t}")
        for metric in results[0]["history"][t]:
            for k, kind in enumerate(("rms", "delta")):
                if t == 0 and kind == "delta":
                    continue
                ys = [r["history"][t][metric][k] for r in results]
                values = "  ".join(f"{y:9.2e}" for y in ys)
                # max/min across sizes: 1 means flat, robust to a degenerate x axis
                spread = max(ys) / max(min(ys), 1e-30)
                print(
                    f"  {metric:>10} {kind:>5}  {values}   "
                    f"slope {slope(xs, ys):+.2f}  max/min {spread:5.2f}"
                )
    print(
        "\nMuon spectral ratio mean/max:",
        "  ".join(f"{r['ratio_mean']:.3f}/{r['ratio_max']:.3f}" for r in results),
    )
    print("train loss at last step:", "  ".join(f"{r['loss']:.3f}" for r in results))


@torch.no_grad()
def check_export(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    # base != size in both width and depth, alpha != 1: every multiplier is exercised
    model = build_model(make_size(512, 4), make_size(256, 2), 2.5, args.seq_len)
    model.eval()
    tokens = torch.randint(VOCAB_SIZE, (2, args.seq_len), device=DEVICE)
    reference = model(input_ids=tokens).logits
    print(f"m_d={model.width_multiplier} m_L={model.depth_multiplier} alpha=2.5")

    plain = model.export_plain_llama().to(DEVICE).eval()
    plain.set_attn_implementation("sdpa")
    exported = plain(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(exported, reference, rtol=1e-4, atol=1e-5)
    print("export: OK")

    compiled = torch.compile(model)
    torch.testing.assert_close(
        compiled(input_ids=tokens).logits, reference, rtol=1e-4, atol=1e-5
    )
    print("compile: OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["export", "width", "depth", "joint"])
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=MODEL_SIZES,
        default=["150M", "300M", "500M", "750M", "1B"],
        help="MODEL_SIZES ladder for the joint check, the first one is the base",
    )
    parser.add_argument(
        "--no-mup", action="store_true", help="control run: every multiplier at 1"
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument(
        "--layers", type=int, default=4, help="depth for the width check"
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument(
        "--width", type=int, default=512, help="width for the depth check"
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    # train.py's defaults, at a constant lr (no warmup) so early updates are full-size
    parser.add_argument("--embedding-multiplier", type=float, default=1.0)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adamw-lr", type=float, default=3e-3)
    parser.add_argument("--muon-weight-decay", type=float, default=0.015)
    parser.add_argument("--adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "export":
        check_export(args)
        return

    if args.mode == "width":
        xs = args.widths
        sizes = [make_size(w, args.layers) for w in xs]
        base = make_size(xs[0], args.layers)
    elif args.mode == "joint":
        sizes = [MODEL_SIZES[name] for name in args.sizes]
        base = sizes[0]
        xs = [non_embedding_millions(size) for size in sizes]
        print("joint sizes:", args.sizes)
    else:
        xs = args.depths
        sizes = [make_size(args.width, d) for d in xs]
        base = make_size(args.width, xs[0])

    results = [run_one(size, None if args.no_mup else base, args) for size in sizes]
    mode = "no-mup" if args.no_mup else "mup"
    report(f"{args.mode} ({mode})", xs, results, args.steps)


if __name__ == "__main__":
    main()
