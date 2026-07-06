from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _package_version(package: str) -> str | None:
    try:
        module = __import__(package)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _command_output(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def inspect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "flash_attn_available": importlib.util.find_spec("flash_attn") is not None,
        "triton_available": importlib.util.find_spec("triton") is not None,
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "gpu": [],
        "cuda_version": None,
        "supported_dtypes": [],
    }

    try:
        import torch

        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["supported_dtypes"] = ["float32"]
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                info["supported_dtypes"].append("bfloat16")
            info["supported_dtypes"].append("float16")
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                info["gpu"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
    except Exception as exc:
        info["torch_inspection_error"] = repr(exc)

    return info


def write_environment_report(output_dir: str | Path) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    info = inspect_environment()
    (output_path / "environment.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    return info
