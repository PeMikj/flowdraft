# Orthrus Qwen3-0.6B training configuration

The reproduced drafter used a frozen `Qwen/Qwen3-0.6B` AR backbone and trained only the added
diffusion attention projections and norms.

```bash
python scripts/train_orthrus_drafter.py \
  --model Qwen/Qwen3-0.6B \
  --dataset-spec yahma/alpaca-cleaned::train \
  --dataset-spec openai/gsm8k:main:train \
  --dataset-spec sahil2801/CodeAlpaca-20k::train \
  --dataset-limit 20000 \
  --text-field text \
  --output orthrus_qwen3_06b_k32_causal_adapter.pt \
  --sequence-length 1024 \
  --context-length 128 \
  --block-size 32 \
  --anchors-per-sequence 8 \
  --num-samples 60000 \
  --batch-size 1 \
  --steps 20000 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --min-learning-rate 2e-5 \
  --warmup-ratio 0.05 \
  --temperature 1.0 \
  --hard-ce-weight 0.0 \
  --model-dtype float16 \
  --device cuda
```

Dataset prompts can be generated with `benchfw.dataset_prompt_builder.build_default_dataset_prompts`,
then benchmarked with:

```bash
python scripts/benchmark_orthrus_drafter.py \
  --checkpoint orthrus_qwen3_06b_k32_causal_adapter.pt \
  --prompts dataset_prompts.jsonl \
  --output-dir outputs/orthrus_qwen3_06b_dataset \
  --max-new-tokens 128 \
  --dtype float32 \
  --require-lossless
```

FP32 is the reference validation mode on T4. FP16 batched verification can accumulate a numerically
different KV cache from token-by-token FP16 AR and is therefore not used for strict token-ID claims.
