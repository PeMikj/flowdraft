from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchfw.env import write_environment_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/env_inspection")
    args = parser.parse_args()
    info = write_environment_report(args.output_dir)
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
