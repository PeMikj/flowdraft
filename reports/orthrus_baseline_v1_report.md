# Orthrus Baseline V1 Report

Date: 2026-07-06

## Scope

This project currently covers inference and benchmarking for:

- AR baseline;
- official Orthrus masked-diffusion decoding;
- exact token-ID losslessness checks;
- acceptance length;
- Tokens Per Forward Pass;
- drafter/verifier forward-pass counts;
- wall-clock throughput.

It does not yet cover drafter training.

## Implementation

Framework files:

- `benchfw/config.py` - benchmark config dataclasses.
- `benchfw/env.py` - runtime environment inspection.
- `benchfw/model_loading.py` - HuggingFace model/tokenizer loading and runtime compatibility patches.
- `benchfw/generation_modes.py` - common generation mode interface, AR mode, Orthrus mode.
- `benchfw/orthrus_instrumentation.py` - Orthrus acceptance/TPF instrumentation.
- `benchfw/result_schema.py` - unified result schema.
- `benchfw/runner.py` - benchmark loop, losslessness verification, summaries.
- `benchfw/dataset_prompt_builder.py` - prompt construction for GSM8K, HumanEval, MATH-500, and MBPP.

The benchmark runner does not depend on Orthrus internals. Orthrus-specific details are isolated in the Orthrus generation mode and instrumentation.

## Partial Benchmark

Prompt suite:

- `prompts/benchmark_prompts_v1.jsonl`
- 24 prompts
- categories: math, code, reasoning, QA, summary, planning
- greedy decoding / temperature 0 behavior

Overall result:

| mode | runs | mean tok/s | mean ms/token | mean acceptance | mean TPF | lossless |
|---|---:|---:|---:|---:|---:|---:|
| AR baseline | 72 | 52.2120 | 20.6806 | n/a | 1.0000 | 100% |
| Orthrus | 72 | 52.3239 | 20.6295 | 3.1347 | 2.0553 | 100% |

Orthrus additional metrics:

- mean drafter forward passes: 24.5000
- mean verifier forward passes: 24.5000
- mean total forward passes: 50.0000
- draft block size: 32

## Dataset Benchmark

Datasets:

| dataset | source | split | selected rows |
|---|---|---|---:|
| GSM8K | `openai/gsm8k`, config `main` | test | 20 |
| HumanEval | `openai/openai_humaneval` | test | 20 |
| MATH-500 | `HuggingFaceH4/MATH-500` | test | 20 |
| MBPP | `google-research-datasets/mbpp`, config `full` | test | 20 |

Overall result:

| mode | runs | mean tok/s | mean ms/token | mean acceptance | mean TPF | lossless |
|---|---:|---:|---:|---:|---:|---:|
| AR baseline | 80 | 61.6997 | 16.7581 | n/a | 1.0000 | 100% |
| Orthrus | 80 | 61.9074 | 16.7380 | 4.0836 | 2.5418 | 100% |

Metrics by dataset:

| dataset | mode | tok/s | ms/token | acceptance | TPF | drafter passes | verifier passes |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | AR | 56.8497 | 18.1730 | n/a | 1.0000 | 0.00 | 126.25 |
| GSM8K | Orthrus | 57.2229 | 18.0547 | 3.6128 | 2.3135 | 28.15 | 28.15 |
| HumanEval | AR | 66.2326 | 15.5446 | n/a | 1.0000 | 0.00 | 83.10 |
| HumanEval | Orthrus | 66.6569 | 15.4721 | 4.6528 | 2.8124 | 15.15 | 15.15 |
| MATH-500 | AR | 65.4589 | 15.6968 | n/a | 1.0000 | 0.00 | 128.00 |
| MATH-500 | Orthrus | 65.6134 | 15.6865 | 4.3831 | 2.6945 | 24.35 | 24.35 |
| MBPP | AR | 58.2577 | 17.6182 | n/a | 1.0000 | 0.00 | 83.75 |
| MBPP | Orthrus | 58.1366 | 17.7390 | 3.6858 | 2.3469 | 18.30 | 18.30 |

## Interpretation

- Orthrus is strictly lossless relative to AR on the tested prompts and dataset subsets.
- Orthrus improves algorithmic TPF.
- Wall-clock throughput was roughly flat on the tested P100 GPU.
- P100 is not representative of the intended modern-GPU deployment environment for this method.

## Next Steps

1. Add training harness for a controlled Orthrus-style masked-diffusion drafter.
2. Add Categorical Flow Map drafter training.
3. Evaluate both drafters with the same frozen AR backbone, data, K, and training budget.
4. Run final benchmarks on modern GPUs.
