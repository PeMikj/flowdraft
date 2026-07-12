# FlowDraft

FlowDraft introduces a Categorical Flow Map (CFM) drafter for lossless parallel decoding in the Orthrus framework. Orthrus accelerates a frozen autoregressive LLM by drafting a block of tokens, then verifying the proposal with the frozen AR head so the final output remains exactly identical to the base model.

The central idea is to replace Orthrus' single-step masked-diffusion drafter with a one- or few-step categorical flow map that produces a more correlated joint proposal over the token block. The intended training objective combines AR-teacher distillation with flow-map consistency, aiming to increase acceptance length and throughput at the same verification cost while preserving strict losslessness.

This repository contains the evaluation infrastructure, Orthrus baseline harness, and an experimental CFM drafter training path needed to measure that goal.

## Install

```bash
git clone https://github.com/PeMikj/flowdraft.git
cd flowdraft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Qwen3-0.6B

The exact training command and benchmark command are in
[`configs/orthrus_qwen3_06b_training.md`](configs/orthrus_qwen3_06b_training.md).

Build the four-dataset prompt subset with:

```bash
python scripts/build_dataset_prompts.py \
  --output dataset_prompts.jsonl \
  --max-per-dataset 20
```

Training produces a diffusion-only adapter checkpoint. Benchmarking loads the frozen Qwen backbone,
applies that adapter, warms up both inference paths, and reports acceptance length, TPF, wall-clock
throughput, forward-pass counts, and exact token-ID matches.

## Current Harness

- standard autoregressive decoding;
- official Orthrus masked-diffusion decoding via `use_diffusion_mode=True`;
- exact token-ID losslessness checks;
- wall-clock throughput and latency;
- Orthrus acceptance length, TPF, drafter/verifier forward-pass metrics;
- prompt and dataset benchmark support.
- experimental CFM drafter training against a frozen AR teacher;
- lossless CFM drafter inference mode with AR verification.

The CFM drafter is trained separately from the frozen AR backbone. During inference it only proposes a block; the AR verifier still determines the final output exactly.

## Current Results

See [reports/orthrus_baseline_v1_report.md](reports/orthrus_baseline_v1_report.md).

Highlights from the current GPU runs:

- Orthrus exact-match losslessness passed on smoke prompts, synthetic benchmark prompts, and dataset subsets.
- Dataset benchmark covered GSM8K, HumanEval, MATH-500, and MBPP subsets.
- Orthrus improved TPF versus AR, while wall-clock throughput on the tested P100 GPU remained roughly flat.

## Structure

- `benchfw/` - benchmark framework and generation modes.
- `configs/` - environment-independent benchmark configs.
- `docs/` - short run instructions.
- `prompts/` - smoke and synthetic prompt suites.
- `scripts/` - local entrypoints for benchmark and environment inspection.
- `scripts/train_cfm_drafter.py` - CFM drafter training entrypoint.
- `scripts/train_orthrus_drafter.py` - Qwen3 Orthrus drafter training entrypoint.
- `scripts/benchmark_orthrus_drafter.py` - AR versus reproduced Orthrus benchmark.
- `reports/` - detailed run reports.

## Kaggle

See [docs/run_kaggle.md](docs/run_kaggle.md).

## Local Checks

```bash
python3 -m py_compile benchfw/*.py scripts/*.py
```
