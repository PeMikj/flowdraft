from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def patch_broken_flex_attention_import() -> None:
    import sys
    import types

    module_name = "torch.nn.attention.flex_attention"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except Exception:
        fake = types.ModuleType(module_name)
        fake._DEFAULT_SPARSE_BLOCK_SIZE = 128
        fake.BlockMask = object
        fake.flex_attention = None
        fake.create_block_mask = None
        sys.modules[module_name] = fake


patch_broken_flex_attention_import()


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def build_dense_dual_pass_mask(
    *,
    batch_size: int,
    diffusion_length: int,
    ar_len: int,
    block_size: int,
    causal_limit: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    q_idx = torch.arange(diffusion_length, device=device)
    kv_idx = torch.arange(ar_len + diffusion_length, device=device)
    is_kv_ar = kv_idx < ar_len
    valid_ar = is_kv_ar[None, None, :] & (kv_idx[None, None, :] <= causal_limit[:, :, None])

    draft_kv_idx = kv_idx - ar_len
    q_block_id = q_idx // block_size
    kv_block_id = draft_kv_idx // block_size
    valid_diffusion = (~is_kv_ar)[None, None, :] & (q_block_id[None, :, None] == kv_block_id[None, None, :])
    valid = valid_ar | valid_diffusion

    mask = torch.full((batch_size, 1, diffusion_length, ar_len + diffusion_length), torch.finfo(dtype).min, device=device, dtype=dtype)
    return mask.masked_fill(valid.unsqueeze(1), 0.0)


def build_dense_causal_mask(
    *,
    position_ids: Tensor,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Build the additive AR mask used for prefill and block verification."""
    key_positions = torch.arange(target_length, device=device)
    valid = key_positions[None, None, :] <= position_ids[:, :, None]
    if attention_mask is not None and attention_mask.ndim == 2:
        valid = valid & attention_mask[:, None, :target_length].bool()
    mask = torch.full(
        (position_ids.shape[0], 1, position_ids.shape[1], target_length),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    return mask.masked_fill(valid.unsqueeze(1), 0.0)


def _import_qwen3() -> dict[str, Any]:
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3Config,
        Qwen3MLP,
        Qwen3PreTrainedModel,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
    )

    return locals()


_qwen3 = _import_qwen3()
Qwen3Attention = _qwen3["Qwen3Attention"]
Qwen3Config = _qwen3["Qwen3Config"]
Qwen3MLP = _qwen3["Qwen3MLP"]
Qwen3PreTrainedModel = _qwen3["Qwen3PreTrainedModel"]
Qwen3RMSNorm = _qwen3["Qwen3RMSNorm"]
Qwen3RotaryEmbedding = _qwen3["Qwen3RotaryEmbedding"]
apply_rotary_pos_emb = _qwen3["apply_rotary_pos_emb"]
BaseModelOutputWithPast = _qwen3["BaseModelOutputWithPast"]
CausalLMOutputWithPast = _qwen3["CausalLMOutputWithPast"]
Cache = _qwen3["Cache"]
DynamicCache = _qwen3["DynamicCache"]
class OrthrusConfig(Qwen3Config):
    model_type = "orthrus"

    def __init__(self, *args: Any, block_size: int | None = None, mask_token_id: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.mask_token_id = mask_token_id


class OrthrusAttention(Qwen3Attention):
    def __init__(self, config: OrthrusConfig, layer_idx: int) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj_diff = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_diff = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_diff = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_diff = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm_diff = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_diff = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        is_diffusion_pass: bool = False,
        causal_limit: Tensor | None = None,
        ar_seq_len: int | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor | None]:
        if not is_diffusion_pass:
            self.is_causal = True
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        if past_key_values is None or ar_seq_len is None:
            raise ValueError("past_key_values and ar_seq_len are required for diffusion pass")
        if causal_limit is None:
            causal_limit = torch.full(
                hidden_states.shape[:2],
                ar_seq_len - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )

        residual_dtype = hidden_states.dtype
        diff_dtype = self.q_proj_diff.weight.dtype
        hidden_states = hidden_states.to(diff_dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings
        cos = cos.to(diff_dtype)
        sin = sin.to(diff_dtype)

        query_states = self.q_norm_diff(self.q_proj_diff(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm_diff(self.k_proj_diff(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj_diff(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        shared_cache = past_key_values.layers[self.layer_idx]
        shared_key_states = shared_cache.keys[:, :, :ar_seq_len, :].to(diff_dtype)
        shared_value_states = shared_cache.values[:, :, :ar_seq_len, :].to(diff_dtype)

        key_states = torch.cat([shared_key_states, key_states], dim=2)
        value_states = torch.cat([shared_value_states, value_states], dim=2)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        dense_mask = build_dense_dual_pass_mask(
            batch_size=hidden_states.shape[0],
            diffusion_length=hidden_states.shape[1],
            ar_len=ar_seq_len,
            block_size=int(self.config.block_size),
            causal_limit=causal_limit,
            dtype=query_states.dtype,
            device=query_states.device,
        )
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + dense_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj_diff(attn_output).to(residual_dtype), None


class OrthrusDecoderLayer(nn.Module):
    def __init__(self, config: OrthrusConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OrthrusAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[Tensor, Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class OrthrusModel(Qwen3PreTrainedModel):
    _no_split_modules = ["OrthrusDecoderLayer"]

    def __init__(self, config: OrthrusConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([OrthrusDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        is_diffusion_pass: bool = False,
        causal_limit: Tensor | None = None,
        ar_seq_len: int | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if is_diffusion_pass:
            causal_mask = attention_mask
        else:
            try:
                from transformers.masking_utils import create_causal_mask

                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            except (ImportError, AttributeError, TypeError):
                target_length = int(cache_position[-1].item()) + 1
                causal_mask = build_dense_causal_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=attention_mask,
                )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                is_diffusion_pass=is_diffusion_pass,
                causal_limit=causal_limit,
                ar_seq_len=ar_seq_len,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class OrthrusLM(Qwen3PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: OrthrusConfig) -> None:
        super().__init__(config)
        self.model = OrthrusModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: Tensor | None = None,
        logits_to_keep: int | Tensor = 0,
        is_diffusion_pass: bool = False,
        causal_limit: Tensor | None = None,
        ar_seq_len: int | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            is_diffusion_pass=is_diffusion_pass,
            causal_limit=causal_limit,
            ar_seq_len=ar_seq_len,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        elif not isinstance(logits_to_keep, int):
            hidden_states = hidden_states[:, logits_to_keep, :]
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(outputs.last_hidden_state,),
        )


def initialize_diffusion_from_ar(model: OrthrusLM) -> None:
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj_diff.load_state_dict(attn.q_proj.state_dict())
        attn.k_proj_diff.load_state_dict(attn.k_proj.state_dict())
        attn.v_proj_diff.load_state_dict(attn.v_proj.state_dict())
        attn.o_proj_diff.load_state_dict(attn.o_proj.state_dict())
        attn.q_norm_diff.load_state_dict(attn.q_norm.state_dict())
        attn.k_norm_diff.load_state_dict(attn.k_norm.state_dict())


def set_trainable_diffusion_only(model: OrthrusLM) -> list[str]:
    trainable: list[str] = []
    for name, param in model.named_parameters():
        is_diff = any(part in name for part in ("q_proj_diff", "k_proj_diff", "v_proj_diff", "o_proj_diff", "q_norm_diff", "k_norm_diff"))
        if is_diff:
            param.data = param.data.float()
        param.requires_grad_(is_diff)
        if is_diff:
            trainable.append(name)
    return trainable


def diffusion_state_dict(model: OrthrusLM) -> dict[str, Tensor]:
    return {name: param.detach().cpu() for name, param in model.state_dict().items() if "_diff" in name}
