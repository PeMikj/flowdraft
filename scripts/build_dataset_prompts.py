from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchfw.dataset_prompt_builder import build_default_dataset_prompts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-dataset", type=int, default=20)
    args = parser.parse_args()
    manifest = build_default_dataset_prompts(args.output, max_per_dataset=args.max_per_dataset)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
