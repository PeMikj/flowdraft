from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from .config import GenerationConfig, ModeConfig
from .orthrus_instrumentation import instrumented_orthrus_generate
from .result_schema import GenerationResult


class GenerationMode(ABC):
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt_id: str,
        prompt: str,
        prompt_metadata: dict[str, Any],
        generation_config: GenerationConfig,
        run_index: int,
    ) -> GenerationResult:
        raise NotImplementedError


def _sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _peak_memory() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        return None
    return None


def _model_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _tokenize(tokenizer: Any, prompt: str, model: Any, generation_config: GenerationConfig) -> dict[str, Any]:
    if generation_config.use_chat_template:
        messages = []
        if generation_config.system_prompt:
            messages.append({"role": "system", "content": generation_config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    return {key: value.to(device) for key, value in encoded.items()}


def _base_generate_kwargs(config: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.eos_token_id is not None:
        kwargs["eos_token_id"] = config.eos_token_id
    if config.pad_token_id is not None:
        kwargs["pad_token_id"] = config.pad_token_id
    return kwargs


def _extract_orthrus_metrics(model: Any, output: Any) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    candidates = [
        getattr(output, "orthus_metrics", None),
        getattr(output, "orthrus_metrics", None),
        getattr(output, "generation_metrics", None),
        getattr(model, "orthus_metrics", None),
        getattr(model, "orthrus_metrics", None),
        getattr(model, "generation_metrics", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            metrics.update(candidate)

    aliases = {
        "acceptance_length": ["acceptance_length", "avg_acceptance_length", "mean_acceptance_length"],
        "acceptance_length_distribution": ["acceptance_length_distribution", "acceptance_lengths"],
        "tpf": ["tpf", "tokens_per_forward_pass"],
        "verifier_forward_passes": ["verifier_forward_passes", "num_verifier_calls"],
        "drafter_forward_passes": ["drafter_forward_passes", "num_drafter_calls"],
        "total_forward_passes": ["total_forward_passes", "num_forward_passes"],
        "draft_block_size": ["draft_block_size", "block_size", "k", "K"],
    }
    normalized: dict[str, Any] = {}
    for normalized_key, keys in aliases.items():
        for key in keys:
            if key in metrics:
                normalized[normalized_key] = metrics[key]
                break
    return normalized


class AutoregressiveMode(GenerationMode):
    def generate(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt_id: str,
        prompt: str,
        prompt_metadata: dict[str, Any],
        generation_config: GenerationConfig,
        run_index: int,
    ) -> GenerationResult:
        import torch

        inputs = _tokenize(tokenizer, prompt, model, generation_config)
        input_length = int(inputs["input_ids"].shape[-1])
        generate_kwargs = _base_generate_kwargs(generation_config)

        _reset_peak_memory()
        _sync_cuda()
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)
        _sync_cuda()
        elapsed = time.perf_counter() - start

        ids = output_ids[0].detach().cpu().tolist()
        generated_only_ids = ids[input_length:]
        return GenerationResult(
            mode_name=self.name,
            prompt_id=prompt_id,
            run_index=run_index,
            prompt_metadata=prompt_metadata,
            generated_token_ids=ids,
            generated_only_token_ids=generated_only_ids,
            decoded_text=tokenizer.decode(ids, skip_special_tokens=False),
            generated_only_text=tokenizer.decode(generated_only_ids, skip_special_tokens=False),
            generation_time_s=elapsed,
            input_length=input_length,
            output_length=len(ids),
            peak_gpu_memory_bytes=_peak_memory(),
            internal_metrics={
                "acceptance_length": None,
                "tpf": 1.0,
                "verifier_forward_passes": max(0, len(ids) - input_length),
                "drafter_forward_passes": 0,
                "total_forward_passes": max(0, len(ids) - input_length),
                "draft_block_size": None,
            },
        )


class OrthrusMode(GenerationMode):
    def generate(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt_id: str,
        prompt: str,
        prompt_metadata: dict[str, Any],
        generation_config: GenerationConfig,
        run_index: int,
    ) -> GenerationResult:
        import torch

        inputs = _tokenize(tokenizer, prompt, model, generation_config)
        input_length = int(inputs["input_ids"].shape[-1])
        generate_kwargs = _base_generate_kwargs(generation_config)
        generate_kwargs.update({key: value for key, value in self.kwargs.items() if key != "block_size"})
        generate_kwargs["use_diffusion_mode"] = True

        _reset_peak_memory()
        _sync_cuda()
        start = time.perf_counter()
        old_block_size = getattr(model.config, "block_size", None)
        with torch.inference_mode():
            if "block_size" in self.kwargs:
                model.config.block_size = int(self.kwargs["block_size"])
            try:
                output_ids, internal_metrics = instrumented_orthrus_generate(
                    model,
                    input_ids=inputs["input_ids"],
                    **generate_kwargs,
                )
            finally:
                if old_block_size is not None:
                    model.config.block_size = old_block_size
        _sync_cuda()
        elapsed = time.perf_counter() - start

        ids = output_ids[0].detach().cpu().tolist()
        generated_only_ids = ids[input_length:]
        return GenerationResult(
            mode_name=self.name,
            prompt_id=prompt_id,
            run_index=run_index,
            prompt_metadata=prompt_metadata,
            generated_token_ids=ids,
            generated_only_token_ids=generated_only_ids,
            decoded_text=tokenizer.decode(ids, skip_special_tokens=False),
            generated_only_text=tokenizer.decode(generated_only_ids, skip_special_tokens=False),
            generation_time_s=elapsed,
            input_length=input_length,
            output_length=len(ids),
            peak_gpu_memory_bytes=_peak_memory(),
            internal_metrics=internal_metrics or _extract_orthrus_metrics(model, output_ids),
        )


def build_mode(config: ModeConfig) -> GenerationMode:
    if config.kind == "autoregressive":
        return AutoregressiveMode(config.name, **config.kwargs)
    if config.kind == "orthrus":
        return OrthrusMode(config.name, **config.kwargs)
    raise ValueError(f"Unsupported generation mode kind: {config.kind}")
