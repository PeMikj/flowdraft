from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .orthrus_model import OrthrusConfig, OrthrusLM, initialize_diffusion_from_ar


def load_orthrus_adapter(model_name: str, checkpoint_path: str | Path, *, dtype: torch.dtype, device: str):
    from transformers import AutoConfig, AutoTokenizer

    payload = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    config_dict = AutoConfig.from_pretrained(model_name, trust_remote_code=True).to_dict()
    config_dict.update(
        block_size=int(payload["block_size"]),
        mask_token_id=int(payload["mask_token_id"]),
        _attn_implementation="eager",
    )
    model = OrthrusLM.from_pretrained(
        model_name,
        config=OrthrusConfig(**config_dict),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    initialize_diffusion_from_ar(model)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def _finish_metrics(metrics: dict[str, Any], generated_tokens: int) -> dict[str, Any]:
    decode_passes = metrics["drafter_forward_passes"] + metrics["verifier_forward_passes"]
    acceptance = metrics["acceptance_length_distribution"]
    metrics["acceptance_length"] = sum(acceptance) / len(acceptance) if acceptance else None
    metrics["decode_forward_passes"] = decode_passes
    metrics["total_forward_passes"] = metrics["initial_forward_passes"] + decode_passes
    metrics["tpf"] = generated_tokens / decode_passes if decode_passes else None
    return metrics


@torch.inference_mode()
def generate_ar(model: OrthrusLM, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None):
    from transformers.cache_utils import DynamicCache

    generated = input_ids.clone()
    cache = DynamicCache(config=model.config)
    positions = torch.arange(generated.shape[1], device=generated.device).unsqueeze(0)
    outputs = model(input_ids=generated, position_ids=positions, past_key_values=cache, use_cache=True)
    next_token = outputs.logits[:, -1].argmax(dim=-1)
    forward_passes = 0
    for _ in range(max_new_tokens):
        generated = torch.cat([generated, next_token[:, None]], dim=1)
        forward_passes += 1
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        positions = torch.tensor([[generated.shape[1] - 1]], device=generated.device)
        outputs = model(input_ids=next_token[:, None], position_ids=positions, past_key_values=cache, use_cache=True)
        next_token = outputs.logits[:, -1].argmax(dim=-1)
    return generated, {
        "acceptance_length": None,
        "tpf": 1.0,
        "drafter_forward_passes": 0,
        "verifier_forward_passes": forward_passes,
        "total_forward_passes": forward_passes,
    }


@torch.inference_mode()
def generate_orthrus(model: OrthrusLM, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None):
    from transformers.cache_utils import DynamicCache

    input_length = input_ids.shape[1]
    max_length = input_length + max_new_tokens
    block_size = int(model.config.block_size)
    mask_token_id = int(model.config.mask_token_id)
    cache = DynamicCache(config=model.config)
    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=input_ids.device
    )
    output_ids[:, :input_length] = input_ids
    metrics = {
        "acceptance_length_distribution": [],
        "draft_block_size": block_size,
        "drafter_forward_passes": 0,
        "verifier_forward_passes": 0,
        "initial_forward_passes": 1,
    }

    positions = torch.arange(input_length, device=input_ids.device).unsqueeze(0)
    outputs = model(input_ids=input_ids, position_ids=positions, past_key_values=cache, use_cache=True)
    start = input_length
    output_ids[:, start] = outputs.logits[:, -1].argmax(dim=-1)
    if eos_token_id is not None and output_ids[0, start].item() == eos_token_id:
        result = output_ids[:, : start + 1]
        return result, _finish_metrics(metrics, result.shape[1] - input_length)

    while start < max_length - 1:
        draft_length = min(block_size, max_length - start)
        draft_input = torch.full((1, draft_length), mask_token_id, dtype=torch.long, device=input_ids.device)
        draft_input[:, 0] = output_ids[:, start]
        positions = torch.arange(start, start + draft_length, device=input_ids.device).unsqueeze(0)
        draft = model(
            input_ids=draft_input,
            position_ids=positions,
            past_key_values=cache,
            use_cache=False,
            is_diffusion_pass=True,
            ar_seq_len=start,
        )
        metrics["drafter_forward_passes"] += 1
        draft_tokens = draft.logits[:, :-1].argmax(dim=-1)
        proposed = torch.cat([output_ids[:, start : start + 1], draft_tokens], dim=1)
        verified = model(
            input_ids=proposed,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            is_diffusion_pass=False,
        )
        metrics["verifier_forward_passes"] += 1
        verifier_tokens = verified.logits.argmax(dim=-1)
        matches = draft_tokens == verifier_tokens[:, :-1]
        accepted = int(matches.cumprod(dim=1).sum(dim=1)[0].item()) if draft_tokens.numel() else 0
        next_token = verifier_tokens[:, accepted]
        accepted_block = proposed[:, : accepted + 1]
        metrics["acceptance_length_distribution"].append(accepted)

        eos = (accepted_block == eos_token_id).nonzero() if eos_token_id is not None else []
        if len(eos):
            eos_offset = int(eos[0, -1].item())
            output_ids[:, start : start + eos_offset + 1] = accepted_block[:, : eos_offset + 1]
            result = output_ids[:, : start + eos_offset + 1]
            return result, _finish_metrics(metrics, result.shape[1] - input_length)

        end = start + accepted + 1
        output_ids[:, start:end] = accepted_block
        start = end
        cache.crop(start)
        if start < max_length:
            output_ids[:, start] = next_token
            if eos_token_id is not None and next_token.item() == eos_token_id:
                result = output_ids[:, : start + 1]
                return result, _finish_metrics(metrics, result.shape[1] - input_length)

    result = output_ids[:, :max_length]
    return result, _finish_metrics(metrics, result.shape[1] - input_length)
