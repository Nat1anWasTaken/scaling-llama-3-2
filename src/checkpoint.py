import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .model import LlamaCausalLM
from .trainer import TrainerState

STEP_DIR_PATTERN = re.compile(r"step-(\d+)")


class PretrainedCheckpointer:
    def __init__(self, output_dir: Path, tokenizer: PreTrainedTokenizerFast) -> None:
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def save(
        self,
        trainer_state: TrainerState,
        model: PreTrainedModel,
        resume_state: dict[str, Any],
    ) -> None:
        step_dir = self.output_dir / f"step-{trainer_state.step}"
        partial_dir = step_dir.with_name(step_dir.name + ".partial")
        shutil.rmtree(partial_dir, ignore_errors=True)
        partial_dir.mkdir(parents=True)

        # HF format for uploading and Flock eval, resume.pt for continuing the run.
        # The HF dir gets the muP/CompleteP multipliers folded into a plain Llama, since a
        # stock loader never runs our forward. resume.pt keeps the unfolded weights
        hf_model = (
            model.export_plain_llama() if isinstance(model, LlamaCausalLM) else model
        )
        hf_model.save_pretrained(partial_dir)
        del hf_model
        self.tokenizer.save_pretrained(partial_dir)
        torch.save(resume_state, partial_dir / "resume.pt")
        (partial_dir / "trainer_state.json").write_text(
            json.dumps(dataclasses.asdict(trainer_state), indent=2)
        )

        # Commit point: only a complete checkpoint ever gets the step-N name
        partial_dir.rename(step_dir)

    def latest(self) -> Path | None:
        """
        The newest complete checkpoint, skipping .partial directories from a crashed save
        """
        if not self.output_dir.is_dir():
            return None

        steps = [
            (int(match.group(1)), path)
            for path in self.output_dir.iterdir()
            if (match := STEP_DIR_PATTERN.fullmatch(path.name)) and path.is_dir()
        ]

        return max(steps)[1] if steps else None
