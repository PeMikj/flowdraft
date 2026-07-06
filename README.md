# FlowDraft

FlowDraft introduces a Categorical Flow Map (CFM) drafter for lossless parallel decoding in the Orthrus framework. Orthrus accelerates a frozen autoregressive LLM by drafting a block of tokens, then verifying the proposal with the frozen AR head so the final output remains exactly identical to the base model.

The central idea is to replace Orthrus' single-step masked-diffusion drafter with a one- or few-step categorical flow map that produces a more correlated joint proposal over the token block. The intended training objective combines AR-teacher distillation with flow-map consistency, aiming to increase acceptance length and throughput at the same verification cost while preserving strict losslessness.

This repository currently contains the evaluation infrastructure and Orthrus baseline harness needed to measure that goal.

## Current Harness

- standard autoregressive decoding;
- official Orthrus masked-diffusion decoding via `use_diffusion_mode=True`;
- exact token-ID losslessness checks;
- wall-clock throughput and latency;
- Orthrus acceptance length, TPF, drafter/verifier forward-pass metrics;
- prompt and dataset benchmark support.

The current code is an inference and evaluation harness. Training a new drafter is a separate next step.

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
- `reports/` - detailed run reports.

## Kaggle

See [docs/run_kaggle.md](docs/run_kaggle.md).

## Local Checks

```bash
python3 -m py_compile benchfw/*.py scripts/*.py
```
