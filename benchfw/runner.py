from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .config import BenchmarkConfig
from .env import write_environment_report
from .generation_modes import GenerationMode, build_mode
from .io_utils import read_prompts, write_jsonl
from .model_loading import load_model_and_tokenizer
from .result_schema import GenerationResult


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def first_difference(left: list[int], right: list[int]) -> int | None:
    for idx, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def verify_losslessness(results: list[GenerationResult]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, GenerationResult]] = defaultdict(dict)
    for result in results:
        by_key[(result.prompt_id, result.run_index)][result.mode_name] = result

    rows: list[dict[str, Any]] = []
    for (prompt_id, run_index), modes in sorted(by_key.items()):
        if "ar_baseline" not in modes:
            continue
        baseline = modes["ar_baseline"]
        for mode_name, result in sorted(modes.items()):
            if mode_name == "ar_baseline":
                continue
            diff = first_difference(baseline.generated_token_ids, result.generated_token_ids)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "run_index": run_index,
                    "baseline_mode": "ar_baseline",
                    "candidate_mode": mode_name,
                    "lossless_match": diff is None,
                    "first_differing_token_position": diff,
                    "baseline_output_length": baseline.output_length,
                    "candidate_output_length": result.output_length,
                    "baseline_decoded_text": baseline.decoded_text,
                    "candidate_decoded_text": result.decoded_text,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_group(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[(record["mode_name"], str(record.get(group_key) or "unknown"))].append(record)

    rows: list[dict[str, Any]] = []
    for (mode_name, group_value), group_records in sorted(by_group.items()):
        throughput = [row["tokens_per_second"] for row in group_records if row["tokens_per_second"] is not None]
        latency = [row["milliseconds_per_token"] for row in group_records if row["milliseconds_per_token"] is not None]
        rows.append(
            {
                "mode_name": mode_name,
                group_key: group_value,
                "runs": len(group_records),
                "mean_throughput_tokens_per_second": mean(throughput) if throughput else None,
                "std_throughput_tokens_per_second": stdev(throughput) if len(throughput) > 1 else None,
                "mean_latency_ms_per_token": mean(latency) if latency else None,
                "mean_generated_tokens": mean(row["generated_tokens"] for row in group_records),
                "mean_acceptance_length": mean(
                    row["acceptance_length"] for row in group_records if row["acceptance_length"] is not None
                )
                if any(row["acceptance_length"] is not None for row in group_records)
                else None,
                "mean_tpf": mean(row["tpf"] for row in group_records if row["tpf"] is not None)
                if any(row["tpf"] is not None for row in group_records)
                else None,
                "mean_verifier_forward_passes": mean(
                    row["verifier_forward_passes"]
                    for row in group_records
                    if row["verifier_forward_passes"] is not None
                )
                if any(row["verifier_forward_passes"] is not None for row in group_records)
                else None,
                "mean_drafter_forward_passes": mean(
                    row["drafter_forward_passes"] for row in group_records if row["drafter_forward_passes"] is not None
                )
                if any(row["drafter_forward_passes"] is not None for row in group_records)
                else None,
            }
        )
    return rows


def summarize(results: list[GenerationResult], lossless_rows: list[dict[str, Any]], env_info: dict[str, Any]) -> list[dict[str, Any]]:
    records = [result.to_record() for result in results if result.run_index >= 0]
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_mode[record["mode_name"]].append(record)

    lossless_by_mode: dict[str, list[bool]] = defaultdict(list)
    for row in lossless_rows:
        lossless_by_mode[row["candidate_mode"]].append(bool(row["lossless_match"]))

    gpu_names = ",".join(str(gpu.get("name")) for gpu in env_info.get("gpu", []))
    rows: list[dict[str, Any]] = []
    for mode_name, mode_records in sorted(by_mode.items()):
        throughput = [row["tokens_per_second"] for row in mode_records if row["tokens_per_second"] is not None]
        latency = [row["milliseconds_per_token"] for row in mode_records if row["milliseconds_per_token"] is not None]
        acceptance = [row["acceptance_length"] for row in mode_records if row["acceptance_length"] is not None]
        tpf = [row["tpf"] for row in mode_records if row["tpf"] is not None]
        match_values = lossless_by_mode.get(mode_name, [])
        rows.append(
            {
                "mode_name": mode_name,
                "runs": len(mode_records),
                "mean_throughput_tokens_per_second": mean(throughput) if throughput else None,
                "std_throughput_tokens_per_second": stdev(throughput) if len(throughput) > 1 else None,
                "mean_latency_ms_per_token": mean(latency) if latency else None,
                "mean_acceptance_length": mean(acceptance) if acceptance else None,
                "mean_tpf": mean(tpf) if tpf else None,
                "lossless_match_rate": mean(match_values) if match_values else (1.0 if mode_name == "ar_baseline" else None),
                "gpu": gpu_names,
                "cuda_version": env_info.get("cuda_version"),
                "torch_version": env_info.get("torch_version"),
                "transformers_version": env_info.get("transformers_version"),
            }
        )
    return rows


def _run_single_mode(
    mode: GenerationMode,
    *,
    model: Any,
    tokenizer: Any,
    prompt_id: str,
    prompt: str,
    prompt_metadata: dict[str, Any],
    config: BenchmarkConfig,
    run_index: int,
) -> GenerationResult:
    set_seed(config.generation.seed)
    return mode.generate(
        model=model,
        tokenizer=tokenizer,
        prompt_id=prompt_id,
        prompt=prompt,
        prompt_metadata=prompt_metadata,
        generation_config=config.generation,
        run_index=run_index,
    )


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_info = write_environment_report(output_dir)
    prompts = read_prompts(config.prompt_file)
    modes = [build_mode(mode_config) for mode_config in config.modes]
    model, tokenizer = load_model_and_tokenizer(config.model)

    results: list[GenerationResult] = []
    for prompt in prompts:
        for mode in modes:
            for _ in range(config.generation.warmup_runs):
                _run_single_mode(
                    mode,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_id=prompt["prompt_id"],
                    prompt=prompt["prompt"],
                    prompt_metadata=prompt["metadata"],
                    config=config,
                    run_index=-1,
                )
            for run_index in range(config.generation.benchmark_runs):
                results.append(
                    _run_single_mode(
                        mode,
                        model=model,
                        tokenizer=tokenizer,
                        prompt_id=prompt["prompt_id"],
                        prompt=prompt["prompt"],
                        prompt_metadata=prompt["metadata"],
                        config=config,
                        run_index=run_index,
                    )
                )

    records = [result.to_record() for result in results]
    lossless_rows = verify_losslessness(results)
    summary_rows = summarize(results, lossless_rows, env_info)

    write_jsonl(output_dir / "benchmark_results.jsonl", records)
    _write_csv(output_dir / "benchmark_results.csv", records)
    write_jsonl(output_dir / "losslessness.jsonl", lossless_rows)
    _write_csv(output_dir / "losslessness.csv", lossless_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "summary_by_dataset.csv", summarize_by_group(records, "dataset"))
    _write_csv(output_dir / "summary_by_category.csv", summarize_by_group(records, "category"))
    _write_csv(output_dir / "summary_by_length_bucket.csv", summarize_by_group(records, "length_bucket"))
    _write_csv(output_dir / "summary_by_block_size.csv", summarize_by_group(records, "block_size"))
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(json.dumps(config, default=lambda obj: obj.__dict__, indent=2))

    return {
        "output_dir": str(output_dir),
        "num_prompts": len(prompts),
        "num_results": len(results),
        "summary": summary_rows,
        "losslessness": lossless_rows,
    }
