import dataclasses
import json
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .trainer import TrainerState


class PretrainedCheckpointer:
    def __init__(self, output_dir: Path, tokenizer: PreTrainedTokenizerFast) -> None:
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def save(self, trainer_state: TrainerState, model: PreTrainedModel) -> None:
        step_dir = self.output_dir / f"step-{trainer_state.step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(step_dir)
        self.tokenizer.save_pretrained(step_dir)
        (step_dir / "trainer_state.json").write_text(
            json.dumps(dataclasses.asdict(trainer_state), indent=2)
        )
