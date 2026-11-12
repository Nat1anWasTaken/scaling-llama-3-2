import math
from typing import Any

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

MUP_CONFIG_KEYS = (
    "base_hidden_size",
    "base_intermediate_size",
    "base_num_hidden_layers",
    "embedding_multiplier",
)


class ScalingLlamaConfig(LlamaConfig):
    def __init__(
        self,
        base_hidden_size: int | None = None,
        base_intermediate_size: int | None = None,
        base_num_hidden_layers: int | None = None,
        embedding_multiplier: float = 1.0,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        # None means "this model is its own base": every multiplier is 1 and we get plain SP
        self.base_hidden_size = base_hidden_size or self.hidden_size
        self.base_intermediate_size = base_intermediate_size or self.intermediate_size
        self.base_num_hidden_layers = base_num_hidden_layers or self.num_hidden_layers
        self.embedding_multiplier = embedding_multiplier


class LlamaCausalLM(LlamaForCausalLM):
    config_class = ScalingLlamaConfig

    def __init__(self, config: ScalingLlamaConfig) -> None:
        # muP width multiplier m_d and CompleteP depth multiplier m_L, both 1 at the base
        self.width_multiplier = config.hidden_size / config.base_hidden_size
        self.depth_multiplier = config.num_hidden_layers / config.base_num_hidden_layers

        super().__init__(config)

        for layer in self.model.layers:
            assert isinstance(layer, LlamaDecoderLayer)
            layer.self_attn.o_proj.register_forward_hook(self._scale_branch)
            layer.mlp.down_proj.register_forward_hook(self._scale_branch)

    def _scale_branch(
        self, module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
    ) -> torch.Tensor:
        return output / self.depth_multiplier

    def initialize_weights(self) -> None:
        # HF's version dispatches every submodel (here self.model, a LlamaModel) to its own
        # generic _init_weights, so ours would only ever see lm_head. Apply ours everywhere
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range

        if module is self.lm_head:
            # tied to the embedding right after init, so its own init never matters
            return
        if isinstance(module, nn.Linear):
            if module.in_features == self.config.hidden_size:
                base_fan_in = self.config.base_hidden_size
            elif module.in_features == self.config.intermediate_size:
                base_fan_in = self.config.base_intermediate_size
            else:
                raise ValueError(f"unexpected fan_in {module.in_features}")
            nn.init.normal_(
                module.weight, std=std * math.sqrt(base_fan_in / module.in_features)
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, LlamaRMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> CausalLMOutputWithPast:
        # alpha_emb: a tunable constant on the input side (Cerebras 10, MiniCPM 12). Passing
        # inputs_embeds makes HF skip its own embedding lookup
        inputs_embeds = (
            self.model.embed_tokens(input_ids) * self.config.embedding_multiplier
        )
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs,
        )
        # muP readout: logits scaled by 1/m_d, applied in the forward pass rather than the
        # weight, since the weight is the tied embedding
        logits = self.lm_head(outputs.last_hidden_state) / self.width_multiplier
        return CausalLMOutputWithPast(logits=logits)

    @torch.no_grad()
    def export_plain_llama(self) -> LlamaForCausalLM:
        """
        Fold the muP/CompleteP multipliers into the weights, giving a plain Llama that
        computes the same function, for the HF upload / flock eval
        """
        plain_config = LlamaConfig(
            **{
                k: v
                for k, v in self.config.to_dict().items()
                if k not in MUP_CONFIG_KEYS
            }
        )
        plain = LlamaForCausalLM(plain_config).to(self.device)
        plain.load_state_dict(self.state_dict())

        # The branches are linear with no bias, so 1/m_L folds into their last weight
        for layer in plain.model.layers:
            layer.self_attn.o_proj.weight.div_(self.depth_multiplier)
            layer.mlp.down_proj.weight.div_(self.depth_multiplier)

        # The embedding is tied, so alpha_emb scaled into it also scales the logits by
        # alpha_emb. Both that and the 1/m_d readout fold into the final RMSNorm gain
        alpha = self.config.embedding_multiplier
        plain.model.embed_tokens.weight.mul_(alpha)
        plain.model.norm.weight.div_(self.width_multiplier * alpha)
        return plain
