from __future__ import annotations

from typing import Any


def instrumented_orthrus_generate(model: Any, input_ids: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from transformers.cache_utils import DynamicCache

    max_new_tokens = kwargs.get("max_new_tokens")
    max_length = kwargs.get("max_length")
    temperature = kwargs.get("temperature", 0.0)
    top_k = kwargs.get("top_k", 20)
    top_p = kwargs.get("top_p", 0.8)
    eos_token_id = kwargs.get("eos_token_id") or getattr(model.config, "eos_token_id", None)
    streamer = kwargs.get("streamer")

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = max_length or (num_input_tokens + max_new_tokens)
    block_size = model.config.block_size
    mask_token_id = model.config.mask_token_id
    past_key_values = DynamicCache(config=model.config)

    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, :num_input_tokens] = input_ids

    if streamer:
        streamer.put(input_ids)

    metrics: dict[str, Any] = {
        "acceptance_length_distribution": [],
        "draft_block_size": int(block_size),
        "drafter_forward_passes": 0,
        "verifier_forward_passes": 0,
        "initial_forward_passes": 0,
    }

    def sample(logits: torch.Tensor):
        if temperature is None or temperature < 1e-5:
            return logits.argmax(dim=-1), None

        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[..., [-1]]] = -float("Inf")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            logits[sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1]), probs

    position_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)
    outputs = model(input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values)
    metrics["initial_forward_passes"] += 1

    start_idx = num_input_tokens
    next_token, _ = sample(outputs.logits[:, -1, :])
    output_ids[:, start_idx] = next_token

    if streamer:
        streamer.put(next_token)
    if next_token.item() == eos_token_id:
        if streamer:
            streamer.end()
        generated = output_ids[:, : start_idx + 1]
        return generated, _finalize_metrics(metrics, generated_tokens=generated.shape[1] - num_input_tokens)

    while start_idx < max_length - 1:
        diff_len = min(block_size, max_length - start_idx)
        diff_block_ids = torch.full((1, diff_len), mask_token_id, dtype=torch.long, device=device)
        diff_block_ids[:, 0] = output_ids[:, start_idx]
        diff_position_ids = torch.arange(start_idx, start_idx + diff_len, device=device).unsqueeze(0)

        diff_outputs = model(
            input_ids=diff_block_ids,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            is_diffusion_pass=True,
            ar_seq_len=start_idx,
        )
        metrics["drafter_forward_passes"] += 1

        if diff_len > 1:
            diff_tokens, diff_probs = sample(diff_outputs.logits[:, :-1, :])
        else:
            diff_tokens, diff_probs = torch.empty((1, 0), dtype=torch.long, device=device), None

        proposed_block = torch.cat([output_ids[:, start_idx : start_idx + 1], diff_tokens], dim=1)

        ar_outputs = model(
            input_ids=proposed_block,
            position_ids=diff_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            is_diffusion_pass=False,
        )
        metrics["verifier_forward_passes"] += 1
        ar_tokens, ar_probs = sample(ar_outputs.logits)

        acceptance_len = 0
        if temperature is None or temperature < 1e-5:
            matches = diff_tokens == ar_tokens[:, :-1]
            acceptance_len = matches.cumprod(dim=1).sum(dim=1)[0].item()
            next_token = ar_tokens[:, acceptance_len]
        else:
            for i in range(diff_tokens.shape[1]):
                q_prob = diff_probs[0, i, diff_tokens[0, i]]
                p_prob = ar_probs[0, i, diff_tokens[0, i]]
                if torch.rand(1, device=device).item() < min(1.0, (p_prob / max(q_prob, 1e-8)).item()):
                    acceptance_len += 1
                else:
                    break

            p_dist = ar_probs[0, acceptance_len]
            if acceptance_len < diff_tokens.shape[1]:
                residual = torch.clamp(p_dist - diff_probs[0, acceptance_len], min=0.0)
                residual_sum = residual.sum()
                next_token = torch.multinomial(residual / residual_sum if residual_sum > 1e-5 else p_dist, 1)
            else:
                next_token = torch.multinomial(p_dist, 1)

        metrics["acceptance_length_distribution"].append(int(acceptance_len))
        end_idx = start_idx + acceptance_len + 1
        accepted_block = proposed_block[:, : acceptance_len + 1]

        eos_positions = (accepted_block == eos_token_id).nonzero()
        if len(eos_positions) > 0:
            eos_offset = eos_positions[0, -1].item()
            output_ids[:, start_idx : start_idx + eos_offset + 1] = accepted_block[:, : eos_offset + 1]
            if streamer:
                streamer.put(accepted_block[:, 1 : eos_offset + 1])
                streamer.end()
            generated = output_ids[:, : start_idx + eos_offset + 1]
            return generated, _finalize_metrics(metrics, generated_tokens=generated.shape[1] - num_input_tokens)

        output_ids[:, start_idx:end_idx] = accepted_block
        if streamer and acceptance_len > 0:
            streamer.put(accepted_block[:, 1:])

        start_idx = end_idx
        past_key_values.crop(start_idx)

        if start_idx < max_length:
            output_ids[:, start_idx] = next_token
            if streamer:
                streamer.put(next_token)

            if next_token.item() == eos_token_id:
                if streamer:
                    streamer.end()
                generated = output_ids[:, : start_idx + 1]
                return generated, _finalize_metrics(metrics, generated_tokens=generated.shape[1] - num_input_tokens)

    if streamer:
        streamer.end()
    generated = output_ids[:, :max_length]
    return generated, _finalize_metrics(metrics, generated_tokens=generated.shape[1] - num_input_tokens)


def _finalize_metrics(metrics: dict[str, Any], generated_tokens: int) -> dict[str, Any]:
    acceptance_lengths = metrics["acceptance_length_distribution"]
    decode_forward_passes = metrics["drafter_forward_passes"] + metrics["verifier_forward_passes"]
    total_forward_passes = metrics["initial_forward_passes"] + decode_forward_passes
    metrics["acceptance_length"] = (
        sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else None
    )
    metrics["generated_tokens_for_tpf"] = int(generated_tokens)
    metrics["total_forward_passes"] = int(total_forward_passes)
    metrics["decode_forward_passes"] = int(decode_forward_passes)
    metrics["tpf"] = generated_tokens / decode_forward_passes if decode_forward_passes else None
    metrics["tpf_including_prefill"] = generated_tokens / total_forward_passes if total_forward_passes else None
    return metrics
