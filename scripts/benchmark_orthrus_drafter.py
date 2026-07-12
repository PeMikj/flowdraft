from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchfw.io_utils import read_prompts  # noqa: E402
from benchfw.orthrus_inference import generate_ar, generate_orthrus, load_orthrus_adapter  # noqa: E402


def chat_input(tokenizer, prompt: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_tensors="pt"
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    return encoded.to(device)


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def summarize(rows: list[dict], group: str | None = None) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], str(row.get(group) or "all"))].append(row)
    output = []
    for (mode, group_value), records in sorted(groups.items()):
        result = {
            "mode": mode,
            "runs": len(records),
            "tokens_per_second": mean(row["tokens_per_second"] for row in records),
            "milliseconds_per_token": mean(row["milliseconds_per_token"] for row in records),
            "tpf": mean(row["tpf"] for row in records),
            "acceptance_length": mean(
                row["acceptance_length"] for row in records if row["acceptance_length"] is not None
            )
            if mode != "ar_baseline"
            else None,
            "drafter_forward_passes": mean(row["drafter_forward_passes"] for row in records),
            "verifier_forward_passes": mean(row["verifier_forward_passes"] for row in records),
            "lossless_match_rate": mean(row["lossless_match"] for row in records),
        }
        if group:
            result[group] = group_value
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--require-lossless", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model, tokenizer = load_orthrus_adapter(args.model, args.checkpoint, dtype=dtype, device=args.device)
    prompts = read_prompts(args.prompts)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup = chat_input(tokenizer, prompts[0]["prompt"], args.device)
    generate_ar(model, warmup, 8, tokenizer.eos_token_id)
    generate_orthrus(model, warmup, 8, tokenizer.eos_token_id)

    rows = []
    for prompt in prompts:
        input_ids = chat_input(tokenizer, prompt["prompt"], args.device)
        mode_outputs = {}
        for mode, generator in (("ar_baseline", generate_ar), ("orthrus_adapter", generate_orthrus)):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output_ids, metrics = generator(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            ids = output_ids[0].cpu().tolist()
            generated = len(ids) - input_ids.shape[1]
            mode_outputs[mode] = ids
            rows.append(
                {
                    "mode": mode,
                    "prompt_id": prompt["prompt_id"],
                    "dataset": prompt["metadata"].get("dataset"),
                    "generated_tokens": generated,
                    "seconds": elapsed,
                    "tokens_per_second": generated / elapsed,
                    "milliseconds_per_token": elapsed * 1000 / generated,
                    "acceptance_length": metrics.get("acceptance_length"),
                    "tpf": metrics.get("tpf"),
                    "drafter_forward_passes": metrics.get("drafter_forward_passes"),
                    "verifier_forward_passes": metrics.get("verifier_forward_passes"),
                    "lossless_match": True,
                    "first_differing_token_position": None,
                }
            )
        difference = first_difference(mode_outputs["ar_baseline"], mode_outputs["orthrus_adapter"])
        rows[-1]["lossless_match"] = difference is None
        rows[-1]["first_differing_token_position"] = difference
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    summary = summarize(rows)
    dataset_summary = summarize(rows, "dataset")
    (output_dir / "benchmark_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True))
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "dataset_summary.csv", dataset_summary)
    print(json.dumps({"summary": summary, "dataset_summary": dataset_summary}, indent=2), flush=True)
    mismatches = [row for row in rows if row["mode"] == "orthrus_adapter" and not row["lossless_match"]]
    if args.require_lossless and mismatches:
        raise RuntimeError(f"strict losslessness failed for {len(mismatches)} prompts")


if __name__ == "__main__":
    main()
