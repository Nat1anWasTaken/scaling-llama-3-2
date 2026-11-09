import math
from pathlib import Path

import numpy
import pyarrow.parquet
import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from .data import Batch, PackedDataLoader, SequencePacker, TokenArray
from .loss import CausalLMLossSum
from .trainer import EvaluationResult, forward_batch, uses_varlen_attention


def load_development_texts(repository_id: str) -> list[str]:
    local_dir = Path(
        snapshot_download(
            repository_id, repo_type="dataset", allow_patterns=["*.parquet"]
        )
    )

    texts: list[str] = []

    for path in sorted(local_dir.rglob("*.parquet")):
        table = pyarrow.parquet.read_table(path, columns=["text"])
        texts.extend(table.column("text").to_pylist())

    return texts


class DevelopmentEvaluator:
    """
    Token-averaged cross entropy on the fineweb-edu dev set, for eval/loss during training.

    The dev documents are tokenized like the training shards ([BOS, text, EOS]) and packed
    the same way, once, at construction. Packing drops the last partial row, so a few
    hundred tokens at the very end of the set are never scored. The official score is
    still Flock's evaluation, this is only the in-training signal.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        repository_id: str,
        loss_fn: CausalLMLossSum,
        seq_len: int,
        micro_batch_size: int,
        max_batches: int | None = None,
    ) -> None:
        texts = load_development_texts(repository_id)
        encoded = tokenizer(
            texts, add_special_tokens=False, return_attention_mask=False
        )

        documents: list[TokenArray] = [
            numpy.array(
                [tokenizer.bos_token_id, *token_ids, tokenizer.eos_token_id],
                dtype=numpy.int32,
            )
            for token_ids in encoded["input_ids"]
        ]

        rows = SequencePacker(documents, seq_len)
        self.batches: list[Batch] = list(PackedDataLoader(rows, micro_batch_size))

        if max_batches is not None:
            self.batches = self.batches[:max_batches]

        if not self.batches:
            raise ValueError("the dev set is too small for one batch")

        self.loss_fn = loss_fn

    @torch.no_grad()
    def __call__(self, model: PreTrainedModel) -> EvaluationResult:
        was_training = model.training
        model.eval()
        device = model.device
        varlen = uses_varlen_attention(model)

        loss_sum = 0.0
        num_tokens = 0

        for batch in self.batches:
            logits, labels = forward_batch(model, batch, device, varlen)
            batch_loss_sum, batch_num_tokens = self.loss_fn(logits, labels)
            del logits
            loss_sum += batch_loss_sum.item()
            num_tokens += int(batch_num_tokens.item())

        if was_training:
            model.train()

        loss = loss_sum / num_tokens
        return EvaluationResult(loss=loss, perplexity=math.exp(loss))
