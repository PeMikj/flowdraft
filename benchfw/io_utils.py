from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_prompts(path: str | Path) -> list[dict[str, Any]]:
    prompt_path = Path(path)
    prompts: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(prompt_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if "prompt" not in item:
            raise ValueError(f"{prompt_path}:{line_number} is missing 'prompt'")
        prompts.append(
            {
                "prompt_id": str(item.get("prompt_id", f"prompt_{line_number:04d}")),
                "prompt": str(item["prompt"]),
                "metadata": dict(item.get("metadata", {}) or {}),
            }
        )
    if not prompts:
        raise ValueError(f"{prompt_path} contains no prompts")
    return prompts


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
