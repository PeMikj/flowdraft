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

All runs use greedy decoding (`temperature=0`) and `K=32`. Each dataset row contains the first 20
examples from its test split. Absolute throughput is not directly comparable across the two model
tables because the 1.7B checkpoint ran on a P100 while the reproduced 0.6B adapter ran on a T4.

### Released Orthrus Qwen3-1.7B checkpoint (P100)

| Dataset | AR tok/s | Orthrus tok/s | Acceptance | TPF | Token match |
|---|---:|---:|---:|---:|---:|
| GSM8K | 56.8497 | 57.2229 | 3.6128 | 2.3135 | 100% |
| HumanEval | 66.2326 | 66.6569 | 4.6528 | 2.8124 | 100% |
| MATH-500 | 65.4589 | 65.6134 | 4.3831 | 2.6945 | 100% |
| MBPP | 58.2577 | 58.1366 | 3.6858 | 2.3469 | 100% |
| **Overall** | **61.6997** | **61.9074** | **4.0836** | **2.5418** | **100%** |

### Reproduced Qwen3-0.6B adapter (T4, FP32 validation)

| Dataset | AR tok/s | Orthrus tok/s | Acceptance | TPF | Token match |
|---|---:|---:|---:|---:|---:|
| GSM8K | 27.9976 | 32.0630 | 1.7523 | 1.3794 | 100% |
| HumanEval | 27.9734 | 36.7441 | 2.2202 | 1.6033 | 100% |
| MATH-500 | 27.8361 | 30.6088 | 1.6373 | 1.3236 | 100% |
| MBPP | 28.0598 | 34.8816 | 2.1187 | 1.5393 | 100% |
| **Overall** | **27.9667** | **33.5744** | **1.9321** | **1.4614** | **100%** |

The reported Qwen3-0.6B results use FP32 validation on T4 and exact token-ID comparison.

Detailed reports:

- [released 1.7B baseline](reports/orthrus_baseline_v1_report.md)
- [reproduced 0.6B adapter](reports/orthrus_qwen3_06b_report.md)

## Structure

- `benchfw/` - benchmark framework and generation modes.
- `configs/` - environment-independent benchmark configs.
- `docs/` - short run instructions.
- `prompts/` - smoke and synthetic prompt suites.
- `scripts/` - local entrypoints for benchmark and environment inspection.
- `scripts/train_orthrus_drafter.py` - Qwen3 Orthrus drafter training entrypoint.
- `scripts/benchmark_orthrus_drafter.py` - AR versus reproduced Orthrus benchmark.
- `benchfw/orthrus_cfm.py` - Categorical Flow Map drafter over the Orthrus diffusion attention and shared AR KV cache.
- `scripts/train_orthrus_cfm.py` - CFM drafter training entrypoint (frozen-AR distillation + flow-map consistency).
- `scripts/benchmark_orthrus_cfm.py` - AR versus CFM drafter benchmark with lossless AR verification.
- `reports/` - detailed run reports.

## Kaggle

See [docs/run_kaggle.md](docs/run_kaggle.md).

## Local Checks

```bash
python3 -m py_compile benchfw/*.py scripts/*.py
```
