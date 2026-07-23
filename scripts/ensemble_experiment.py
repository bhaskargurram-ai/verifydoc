#!/usr/bin/env python
"""Real-data ensemble experiment: does adjudicating two extractors beat each alone?

Runs each extractor on the same dataset slice, grounds its predictions, and
adjudicates per field (``verifydoc.agents.adjudicate``). Reports micro P/R/F1 for
every single extractor vs the ensemble, plus the fused hallucination/omission
rates — the paper's multi-extractor result.

    python scripts/ensemble_experiment.py --dataset cord --limit 60 \
        --extractors rapidocr,paddleocr-vl --out results/ensemble-cord.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifydoc.adapters import get_adapter
from verifydoc.agents import adjudicate
from verifydoc.eval.extraction import score_fields
from verifydoc.grounding import ground_predictions


def load_bench(dataset: str, split: str | None, limit: int):
    if dataset == "cord":
        from benchmark.datasets import cord

        return cord.load(split=split or "validation", limit=limit, with_images=True)
    if dataset == "funsd":
        from benchmark.datasets import funsd

        return funsd.load(split=split or "testing", limit=limit)
    raise SystemExit(f"unsupported dataset {dataset!r} (use cord|funsd)")


def micro(reports: list) -> dict:
    nc = sum(r.n_correct for r in reports)
    npd = sum(r.n_predicted for r in reports)
    ng = sum(r.n_gold for r in reports)
    nh = sum(len(r.hallucinated_paths) for r in reports)
    p = nc / npd if npd else 0.0
    r = nc / ng if ng else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "hallucination_rate": round(nh / npd, 4) if npd else 0.0,
        "n_correct": nc,
        "n_pred": npd,
        "n_gold": ng,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="cord")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--extractors", default="rapidocr,paddleocr-vl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = [n.strip() for n in args.extractors.split(",") if n.strip()]
    adapters = [get_adapter(n) for n in names]
    bench = load_bench(args.dataset, args.split, args.limit)

    per_reports: dict[str, list] = {n: [] for n in names}
    ens_reports: list = []
    n_docs = 0
    for item in bench:
        n_docs += 1
        grounded = {}
        for n, a in zip(names, adapters):
            preds = ground_predictions(a.extract(item.doc, item.schema), item.doc)
            grounded[n] = preds
            per_reports[n].append(score_fields(preds, item.golds))
        fused = adjudicate([grounded[n] for n in names], names, n_total=len(names))
        ens_reports.append(score_fields(fused, item.golds))
        if n_docs % 20 == 0:
            print(f"...{n_docs} docs", flush=True)

    result = {
        "dataset": args.dataset,
        "n_docs": n_docs,
        "extractors": names,
        "single": {n: micro(per_reports[n]) for n in names},
        "ensemble": micro(ens_reports),
    }
    best_single_f1 = max(result["single"][n]["f1"] for n in names)
    result["ensemble_gain_f1_vs_best_single"] = round(result["ensemble"]["f1"] - best_single_f1, 4)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
