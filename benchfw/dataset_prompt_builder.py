from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DATASET_SPECS = [
    {
        "dataset_name": "gsm8k",
        "loader": ("openai/gsm8k", "main", "test"),
        "prompt_field": "question",
        "prefix": "Solve the following grade-school math problem. Give the final answer clearly.",
        "category": "math",
    },
    {
        "dataset_name": "humaneval",
        "loader": ("openai/openai_humaneval", None, "test"),
        "prompt_field": "prompt",
        "prefix": "Complete the following Python programming problem. Return code only.",
        "category": "code",
    },
    {
        "dataset_name": "math500",
        "loader": ("HuggingFaceH4/MATH-500", None, "test"),
        "prompt_field": "problem",
        "prefix": "Solve the following mathematics problem. Give the final answer clearly.",
        "category": "math",
    },
    {
        "dataset_name": "mbpp",
        "loader": ("google-research-datasets/mbpp", "full", "test"),
        "prompt_field": "text",
        "prefix": "Solve the following Python programming problem. Return code only.",
        "category": "code",
    },
]


def build_default_dataset_prompts(output_path: str | Path, max_per_dataset: int = 20) -> dict[str, Any]:
    from datasets import load_dataset

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "output_path": str(output),
        "max_per_dataset": max_per_dataset,
        "datasets": [],
    }

    for spec in DATASET_SPECS:
        hf_name, hf_config, split = spec["loader"]
        if hf_config is None:
            dataset = load_dataset(hf_name, split=split)
        else:
            dataset = load_dataset(hf_name, hf_config, split=split)

        count = min(max_per_dataset, len(dataset))
        manifest["datasets"].append(
            {
                "dataset_name": spec["dataset_name"],
                "hf_name": hf_name,
                "hf_config": hf_config,
                "split": split,
                "available_rows": len(dataset),
                "selected_rows": count,
            }
        )

        for idx, item in enumerate(dataset.select(range(count))):
            prompt = _format_prompt(spec, item)
            rows.append(
                {
                    "prompt_id": f"{spec['dataset_name']}_{idx:04d}",
                    "prompt": prompt,
                    "metadata": {
                        "dataset": spec["dataset_name"],
                        "category": spec["category"],
                        "length_bucket": _length_bucket(prompt),
                        "source_index": idx,
                    },
                }
            )

    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest["total_prompts"] = len(rows)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def maybe_build_dataset_prompts_from_env() -> dict[str, Any] | None:
    output_path = os.environ.get("FLOWDRAFT_DATASET_PROMPT_FILE")
    if not output_path:
        return None
    max_per_dataset = int(os.environ.get("FLOWDRAFT_MAX_PER_DATASET", "20"))
    return build_default_dataset_prompts(output_path, max_per_dataset=max_per_dataset)


def _format_prompt(spec: dict[str, Any], item: dict[str, Any]) -> str:
    prompt_body = str(item[spec["prompt_field"]]).strip()
    return f"{spec['prefix']}\n\n{prompt_body}\n\n/no_think"


def _length_bucket(prompt: str) -> str:
    words = len(prompt.split())
    if words < 80:
        return "short"
    if words < 220:
        return "medium"
    return "long"
