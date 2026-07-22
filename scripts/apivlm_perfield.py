#!/usr/bin/env python
"""Real frontier-VLM per-field run: dump signals, then run the method on them.

Unlike the aggregate benchmark row, this persists *per-field* signals from a
real API-VLM (verbalized confidence, grounding support, correctness) so the
grounding-conditioned conformal method can be evaluated on genuine model
outputs rather than a simulation. API calls are parallelized.

Outputs:
  data/apivlm_perfield.json                 — raw per-field records (cached)
  paper/generated/grouped_conformal_apivlm.md — method vs pooled on real VLM data

Env: ANTHROPIC_API_KEY (or OPENAI_API_KEY with --provider openai).
Usage: python scripts/apivlm_perfield.py [--limit 150] [--provider anthropic]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.adapters.api_vlm import APIVLMAdapter  # noqa: E402
from verifydoc.calibration import ConformalAbstention, GroupConformalAbstention  # noqa: E402
from verifydoc.calibration.splits import split_calibration  # noqa: E402
from verifydoc.confidence import verbalized_confidence  # noqa: E402
from verifydoc.eval.extraction import score_fields  # noqa: E402
from verifydoc.grounding import ground_predictions  # noqa: E402

ALPHAS = [0.02, 0.05, 0.10]
N_SPLITS = 40
CACHE = Path("data/apivlm_perfield.json")


def _one_doc(adapter, item):
    preds = ground_predictions(adapter.extract(item.doc, item.schema), item.doc)
    report = score_fields(preds, item.golds)
    by_path = {p.path: p for p in preds}
    out = []
    for fs in report.field_scores:
        p = by_path[fs.path]
        verb = verbalized_confidence(p)
        out.append(
            {
                "doc_id": item.doc.doc_id,
                "verbalized": float(verb) if verb is not None else 0.9,
                "grounded": p.grounding is not None,
                "support": float(p.grounding.support) if p.grounding else 0.0,
                "correct": int(fs.correct),
            }
        )
    return out


def collect(limit: int, provider: str) -> list[dict]:
    if CACHE.exists():
        print(f"using cached {CACHE}")
        return json.loads(CACHE.read_text(encoding="utf-8"))
    from benchmark.datasets import cord

    bench = cord.load(split="validation", limit=limit)
    adapter = APIVLMAdapter(provider=provider)
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_one_doc, adapter, item): item.doc.doc_id for item in bench}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                records.extend(fut.result())
            except Exception as exc:  # one bad doc shouldn't kill the run
                print(f"  doc {futures[fut]} failed: {str(exc)[:80]}")
            if i % 20 == 0:
                print(f"  {i}/{len(bench)} docs done")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(records), encoding="utf-8")
    print(f"dumped {len(records)} field records -> {CACHE}")
    return records


def _grouped_of(support: float, min_support: float = 0.5) -> str:
    return "grounded" if support >= min_support else "ungrounded"


def evaluate(records: list[dict], rows: list[dict]) -> None:
    from verifydoc.types import FieldPrediction, Grounding

    doc_ids = np.array([r["doc_id"] for r in records])
    conf = np.array([r["verbalized"] for r in records])
    correct = np.array([r["correct"] for r in records], dtype=float)
    preds = [
        FieldPrediction(
            path="f",
            value="v",
            confidence=r["verbalized"],
            grounding=Grounding(page=0, support=r["support"]) if r["grounded"] else None,
        )
        for r in records
    ]
    uniq = list(dict.fromkeys(doc_ids.tolist()))
    frac_grnd = float(np.mean([r["grounded"] for r in records]))
    acc = {a: {"cp": [], "cg": [], "rg": []} for a in ALPHAS}
    for s in range(N_SPLITS):
        cal_docs = set(split_calibration(uniq, 0.5, seed=s)[0])
        cal = np.array([d in cal_docs for d in doc_ids])
        test = ~cal
        cal_preds = [p for p, m in zip(preds, cal) if m]
        test_preds = [p for p, m in zip(preds, test) if m]
        for alpha in ALPHAS:
            pooled = ConformalAbstention(alpha=alpha).fit(conf[cal].tolist(), correct[cal].tolist())
            grouped = GroupConformalAbstention(alpha=alpha).fit(cal_preds, correct[cal].tolist())
            pm = pooled.accept(conf[test])
            gm = grouped.accept(test_preds)
            acc[alpha]["cp"].append(float(pm.mean()))
            acc[alpha]["cg"].append(float(gm.mean()))
            if gm.any():
                acc[alpha]["rg"].append(float(1.0 - correct[test][gm].mean()))
    for alpha in ALPHAS:
        cp, cg = float(np.mean(acc[alpha]["cp"])), float(np.mean(acc[alpha]["cg"]))
        rg = float(np.mean(acc[alpha]["rg"])) if acc[alpha]["rg"] else 0.0
        rows.append(
            {
                "n_fields": len(records),
                "frac_grnd": frac_grnd,
                "alpha": alpha,
                "cov_pooled": cp,
                "cov_grounded": cg,
                "cov_lift": cg - cp,
                "risk_grounded": rg,
                "held": rg <= alpha + 0.01,
            }
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--provider", default="anthropic")
    args = ap.parse_args()

    records = collect(args.limit, args.provider)
    acc = float(np.mean([r["correct"] for r in records]))
    grnd = float(np.mean([r["grounded"] for r in records]))
    print(f"real VLM per-field: {len(records)} fields, accuracy {acc:.2f}, grounded {grnd:.2f}")

    rows: list[dict] = []
    evaluate(records, rows)
    out = Path("paper/generated/grouped_conformal_apivlm.md")
    cols = list(rows[0].keys())
    lines = [
        "# Grounding-conditioned vs pooled conformal on REAL frontier-VLM outputs",
        "",
        "Per-field verbalized confidence + grounding from claude-sonnet-5 on CORD;",
        "40-split marginal evaluation. Genuine model outputs, not simulation.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += [
        "| "
        + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
        + " |"
        for r in rows
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in rows:
        print(
            f"  a={r['alpha']:.2f}  pooled={r['cov_pooled']:.3f} grounded={r['cov_grounded']:.3f}  "
            f"lift=+{r['cov_lift']:.3f}  risk={r['risk_grounded']:.3f}  held={r['held']}"
        )


if __name__ == "__main__":
    main()
