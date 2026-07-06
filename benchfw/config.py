from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    tokenizer_name_or_path: str | None = None
    dtype: str = "bfloat16"
    device_map: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 128
    batch_size: int = 1
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    seed: int = 20260705
    warmup_runs: int = 1
    benchmark_runs: int = 3
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    use_chat_template: bool = False
    system_prompt: str | None = None


@dataclass(frozen=True)
class ModeConfig:
    name: str
    kind: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkConfig:
    experiment_name: str
    environment: str
    output_dir: str
    prompt_file: str
    model: ModelConfig
    generation: GenerationConfig
    modes: list[ModeConfig]


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return value


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text()) or {}
    root = _require_mapping(raw, "config")

    model_raw = _require_mapping(root.get("model", {}), "model")
    gen_raw = _require_mapping(root.get("generation", {}), "generation")
    modes_raw = root.get("modes", [])
    if not isinstance(modes_raw, list) or not modes_raw:
        raise ValueError("config must define at least one generation mode")

    modes = []
    for mode in modes_raw:
        mode_map = _require_mapping(mode, "mode")
        modes.append(
            ModeConfig(
                name=str(mode_map["name"]),
                kind=str(mode_map["kind"]),
                kwargs=dict(mode_map.get("kwargs", {}) or {}),
            )
        )

    return BenchmarkConfig(
        experiment_name=str(root["experiment_name"]),
        environment=str(root["environment"]),
        output_dir=str(root["output_dir"]),
        prompt_file=str(root["prompt_file"]),
        model=ModelConfig(**model_raw),
        generation=GenerationConfig(**gen_raw),
        modes=modes,
    )
