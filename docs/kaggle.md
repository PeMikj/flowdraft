# Running Benchmarks on Kaggle

This repository does not store Kaggle credentials, generated Kaggle bundles, kernel outputs, or private kernel identifiers.

Use this guide as a safe template for running the benchmark on Kaggle.

## 1. Create a Kaggle Notebook

Create a new private Kaggle notebook with:

- GPU enabled;
- Internet enabled if the notebook needs to download HuggingFace models or datasets;
- Python environment.

Record the actual GPU from the notebook output. Benchmark results should always be grouped by GPU model.

## 2. Install Runtime Dependencies

For older Kaggle GPUs such as P100, a compatible PyTorch stack may be required. For newer GPUs, use the official environment preferred by the model you are testing.

Example dependency block:

```python
import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "transformers",
    "accelerate",
    "safetensors",
    "pyyaml",
    "datasets",
])
```

If the allocated GPU is old and the preinstalled PyTorch build does not support it, install a compatible PyTorch build before installing the remaining packages.

## 3. Add the Repository Code

Upload or clone the repository into the notebook environment.

The benchmark code expects the repository root to be importable:

```python
import sys
from pathlib import Path

repo_root = Path("/kaggle/working/flowdraft")
sys.path.insert(0, str(repo_root))
```

Do not commit or publish:

- Kaggle API tokens;
- `kaggle.json`;
- notebook output archives;
- generated model weights;
- private dataset paths;
- private kernel URLs.

## 4. Prepare a Config

Configs are normal YAML files. A minimal benchmark config needs:

- experiment name;
- output directory;
- prompt file;
- model name/path;
- generation settings;
- generation modes.

Example mode section:

```yaml
modes:
  - name: ar_baseline
    kind: autoregressive
    kwargs: {}
  - name: orthrus_official
    kind: orthrus
    kwargs: {}
```

Use deterministic generation for losslessness checks:

```yaml
generation:
  do_sample: false
  temperature: null
  max_new_tokens: 128
  warmup_runs: 1
  benchmark_runs: 1
  use_chat_template: true
```

## 5. Build Dataset Prompts on Kaggle

For GSM8K, HumanEval, MATH-500, and MBPP, prompts can be generated inside the Kaggle notebook:

```python
from benchfw.dataset_prompt_builder import build_default_dataset_prompts

build_default_dataset_prompts(
    "/kaggle/working/dataset_prompts.jsonl",
    max_per_dataset=20,
)
```

This creates:

- `dataset_prompts.jsonl`;
- `dataset_prompts.manifest.json`.

The manifest records which HuggingFace datasets, splits, and row counts were used.

## 6. Run the Benchmark

```python
from benchfw.config import load_config
from benchfw.runner import run_benchmark

config = load_config("configs/your_config.yaml")
result = run_benchmark(config)
print(result)
```

All outputs should be written under `/kaggle/working/<experiment_name>/`.

Expected output files:

- `environment.json`
- `benchmark_results.jsonl`
- `benchmark_results.csv`
- `losslessness.jsonl`
- `losslessness.csv`
- `summary.csv`
- `summary.json`
- `summary_by_dataset.csv`
- `summary_by_category.csv`
- `summary_by_length_bucket.csv`
- `summary_by_block_size.csv`
- `run_config.json`

## 7. Required Metrics

For AR and Orthrus-style modes, collect:

- wall-clock generation time;
- generated tokens;
- tokens/sec;
- milliseconds/token;
- peak GPU memory;
- lossless match rate.

For drafter-based modes, also collect:

- mean acceptance length;
- acceptance length distribution;
- Tokens Per Forward Pass;
- drafter forward passes;
- verifier forward passes;
- total forward passes;
- draft block size.

## 8. Reporting Rules

Always report:

- GPU model;
- CUDA version;
- PyTorch version;
- Transformers version;
- dataset sources and splits;
- number of prompts;
- generation settings;
- exact losslessness match rate.

Do not average results across different GPU models unless the table clearly separates them.

## 9. Safety Checklist Before Publishing

Before publishing repository changes, verify that the commit does not contain:

- `kaggle.json`;
- API keys or tokens;
- private kernel IDs;
- private usernames;
- downloaded Kaggle outputs;
- generated bundle directories;
- model checkpoint files.

Useful local checks:

```bash
git status --short
git grep -n -I -E 'api[_-]?key|secret|password|credential|Authorization|Bearer|kaggle.json' -- .
```
