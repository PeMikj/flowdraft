# Orthrus Qwen3-0.6B Reproduction

## Training

- backbone: `Qwen/Qwen3-0.6B`, frozen;
- trainable parameters: diffusion Q/K/V/O projections and Q/K norms;
- block size: 32;
- packed sequence length: 1024;
- anchors per sequence: 8;
- 20,000 microsteps, gradient accumulation 8 (2,500 optimizer updates);
- soft forward KL, no hard-label CE;
- data: Alpaca Cleaned, GSM8K train, and CodeAlpaca;
- hardware: NVIDIA T4.

## Internal Prompt Benchmark

The 24-prompt greedy benchmark used one warmup for both paths and generated up to 64 tokens.

| mode | tok/s | ms/token | acceptance | TPF | lossless |
|---|---:|---:|---:|---:|---:|
| AR | 24.7809 | 40.3722 | n/a | 1.0000 | 100% |
| Orthrus 0.6B | 28.7691 | 35.3471 | 1.7567 | 1.3815 | 100% |

## Dataset Subset Benchmark

The benchmark selected the first 20 test rows from GSM8K, HumanEval, MATH-500, and MBPP and
generated up to 128 tokens at temperature 0.

The strict validation run used FP32 on T4.

| dataset | AR tok/s | Orthrus tok/s | acceptance | TPF | strict token match |
|---|---:|---:|---:|---:|---:|
| GSM8K | 27.9976 | 32.0630 | 1.7523 | 1.3794 | 100% |
| HumanEval | 27.9734 | 36.7441 | 2.2202 | 1.6033 | 100% |
| MATH-500 | 27.8361 | 30.6088 | 1.6373 | 1.3236 | 100% |
| MBPP | 28.0598 | 34.8816 | 2.1187 | 1.5393 | 100% |
| Overall | 27.9667 | 33.5744 | 1.9321 | 1.4614 | 100% |

An FP16 diagnostic run matched 70 of 80 generations. FP32 matched all 80, confirming that the earlier
divergence came from accumulated floating-point differences between batched verification and
token-by-token AR rather than the greedy consensus algorithm. Use `--require-lossless` for reported
validation runs.
