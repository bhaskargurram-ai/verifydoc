"""VerifyDocBench leaderboard: rank extractors on dimensions competitors don't
report — omission vs hallucination, error-ranking AUROC, grounding gap, and
calibration (ECE) — from the per-run ``summary.json`` files the harness writes.

    from verifydoc.eval.leaderboard import leaderboard_table, load_summaries
    print(leaderboard_table(load_summaries(Path("results").glob("*/summary.json"))))

The point of the leaderboard is the framing: every vendor sells a headline
"accuracy", but none publishes whether their confidence is *calibrated* or how
much it *silently hallucinates*. This scores exactly that.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# (column key, extraction path into the summary) — order defines the table.
_COLUMNS = ("dataset", "extractor", "F1", "exact", "halluc", "best_AUROC", "grounding_gap", "ECE")


def leaderboard_row(summary: dict[str, Any]) -> dict[str, Any]:
    """One leaderboard row from a harness ``run_benchmark`` summary dict."""
    ex = summary.get("extraction", {})
    gr = summary.get("grounding", {})
    cov = summary.get("best_coverage", {})
    ece = summary.get("best_ece", {})
    return {
        "dataset": summary.get("dataset", "?"),
        "extractor": summary.get("extractor", "?"),
        "F1": round(float(ex.get("f1", 0.0)), 3),
        "exact": round(float(ex.get("exact_match", 0.0)), 3),
        "halluc": round(float(ex.get("hallucination_rate", 0.0)), 3),
        "best_AUROC": round(float(cov.get("auroc", 0.0)), 3),
        "grounding_gap": round(float(gr.get("grounding_gap", 0.0)), 3),
        "ECE": round(float(ece.get("ece", 0.0)), 4),
    }


def leaderboard_table(summaries: Sequence[dict[str, Any]], sort_key: str = "best_AUROC") -> str:
    """Render a markdown leaderboard, best error-ranking (AUROC) first."""
    rows = [leaderboard_row(s) for s in summaries]
    rows.sort(key=lambda r: float(r.get(sort_key, 0.0)), reverse=True)
    lines = [
        "# VerifyDocBench leaderboard",
        "",
        "Extractors scored by the trust dimensions parsers don't report: error-",
        "ranking `best_AUROC`, `grounding_gap` (grounded vs ungrounded correctness),",
        "calibration `ECE`, and the `halluc`(ination) rate — alongside F1/exact.",
        "Regenerate with `python scripts/leaderboard.py`.",
        "",
        "| " + " | ".join(_COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    lines += ["| " + " | ".join(str(r[c]) for c in _COLUMNS) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def load_summaries(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load harness ``summary.json`` files (skips unreadable/invalid ones)."""
    out: list[dict[str, Any]] = []
    for p in paths:
        try:
            out.append(json.loads(Path(p).read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out
