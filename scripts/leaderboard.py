#!/usr/bin/env python
"""Generate the VerifyDocBench leaderboard from harness summary.json files.

    python scripts/leaderboard.py [results_dir] [--out results/LEADERBOARD.md]

Reads ``<results_dir>/*/summary.json`` (written by every `run_benchmark`), ranks
extractors by error-ranking AUROC + trust dimensions, and writes a markdown
leaderboard. With no runs present it prints a header and a note.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from verifydoc.eval.leaderboard import leaderboard_table, load_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--out", default=None, help="write here (else stdout)")
    args = parser.parse_args()

    summaries = load_summaries(Path(args.results_dir).glob("*/summary.json"))
    table = leaderboard_table(summaries)
    if not summaries:
        table += "\n_No runs found — generate some with `make results` first._\n"
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"wrote {args.out} ({len(summaries)} runs)")
    else:
        print(table)


if __name__ == "__main__":
    main()
