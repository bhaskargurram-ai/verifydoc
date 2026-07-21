#!/usr/bin/env python
"""Thin CLI over verifydoc.eval.harness — regenerates every table/figure.

Usage: python scripts/run_benchmark.py --config configs/demo.yaml --out paper/generated
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `benchmark`

from verifydoc.eval.harness import run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--out", default="paper/generated")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summary = run_benchmark(cfg, args.out)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
