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

| dataset | AR tok/s | Orthrus tok/s | acceptance | TPF | strict token match |
|---|---:|---:|---:|---:|---:|
| GSM8K | 24.7430 | 29.6385 | 1.7780 | 1.3928 | 75% |
| HumanEval | 24.8156 | 33.6082 | 2.2214 | 1.6033 | 95% |
| MATH-500 | 23.6689 | 26.8725 | 1.6220 | 1.3165 | 85% |
| MBPP | 24.7678 | 31.8111 | 2.1187 | 1.5393 | 95% |
| Overall | 24.4988 | 30.4825 | 1.9350 | 1.4630 | 87.5% |

The dataset-subset throughput is preliminary. Batched FP16 verification diverged from token-by-token
AR on 10 of 80 generations, so it must not be presented as a strict lossless result. The benchmark
retains exact token-ID checks to expose this condition rather than hiding it.
