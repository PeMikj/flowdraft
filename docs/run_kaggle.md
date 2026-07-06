# Kaggle Run

## Notebook Setup

Create a Kaggle notebook with GPU and Internet enabled.

Clone the repository:

```bash
git clone https://github.com/PeMikj/flowdraft.git /kaggle/working/flowdraft
cd /kaggle/working/flowdraft
```

Install dependencies:

```bash
pip install -q transformers accelerate safetensors pyyaml datasets
```

If the default PyTorch build does not work with the allocated GPU, install a compatible PyTorch build first, then rerun the dependency install.

## Dataset Prompts

```python
from benchfw.dataset_prompt_builder import build_default_dataset_prompts

build_default_dataset_prompts(
    "/kaggle/working/dataset_prompts.jsonl",
    max_per_dataset=20,
)
```

## Config

Create a YAML config in `/kaggle/working/config.yaml`:

```yaml
experiment_name: orthrus_dataset_benchmark
environment: kaggle
output_dir: /kaggle/working/orthrus_dataset_benchmark
prompt_file: /kaggle/working/dataset_prompts.jsonl

model:
  name_or_path: chiennv/Orthrus-Qwen3-1.7B
  tokenizer_name_or_path: chiennv/Orthrus-Qwen3-1.7B
  dtype: float16
  device_map: auto
  attn_implementation: null
  trust_remote_code: true

generation:
  max_new_tokens: 128
  batch_size: 1
  do_sample: false
  temperature: null
  top_p: null
  seed: 20260705
  warmup_runs: 1
  benchmark_runs: 1
  eos_token_id: null
  pad_token_id: null
  use_chat_template: true
  system_prompt: null

modes:
  - name: ar_baseline
    kind: autoregressive
    kwargs: {}
  - name: orthrus_official
    kind: orthrus
    kwargs: {}
```

## Run

```python
from benchfw.config import load_config
from benchfw.runner import run_benchmark

result = run_benchmark(load_config("/kaggle/working/config.yaml"))
print(result)
```

## Outputs

Results are written to `output_dir`:

- `environment.json`
- `benchmark_results.csv`
- `benchmark_results.jsonl`
- `losslessness.csv`
- `losslessness.jsonl`
- `summary.csv`
- `summary_by_dataset.csv`
- `summary_by_category.csv`
- `summary_by_length_bucket.csv`
- `summary_by_block_size.csv`
- `run_config.json`
