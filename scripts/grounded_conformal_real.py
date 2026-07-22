#!/usr/bin/env python
"""Grounding-conditioned conformal on REAL documents at scale.

Companion to the controlled study (``grouped_conformal_experiment.py``): here
the population is real --- CORD train receipts (~400) and full FUNSD
(~194 forms), thousands of fields with real text layers and real grounding.

A realistic extractor (seeded corruption/omission/hallucination) runs over the
real documents; each value is grounded on the actual page (numeric-aware
matching), and the *accept score is the model's verbalized self-confidence*,
which is inflated and near-uninformative --- exactly the frontier-VLM regime.
Provenance (grounded vs ungrounded) is the orthogonal signal. We compare pooled
split-conformal against grounding-conditioned conformal at matched target risk.

Writes paper/generated/grouped_conformal_real.{md,tex}. Offline (one dataset
download, then cached); deterministic.

Usage: python scripts/grounded_conformal_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.adapters.mock import MockAdapter  # noqa: E402
from verifydoc.calibration import ConformalAbstention, GroupConformalAbstention  # noqa: E402
from verifydoc.calibration.splits import split_calibration  # noqa: E402
from verifydoc.confidence import consensus, verbalized_confidence  # noqa: E402
from verifydoc.eval.extraction import score_fields  # noqa: E402
from verifydoc.grounding import ground_predictions  # noqa: E402

ALPHAS = [0.02, 0.05, 0.10]
K = 3
SEED = 7
N_SPLITS = 40  # conformal's guarantee is marginal (in expectation over splits)
ERROR_RATE, OMIT_RATE, HALLUCINATE_RATE = 0.30, 0.05, 0.10


def collect(bench, seed=SEED):
    adapter = MockAdapter(
        gold={item.doc.doc_id: item.golds for item in bench},
        error_rate=ERROR_RATE,
        omit_rate=OMIT_RATE,
        hallucinate_rate=HALLUCINATE_RATE,
        seed=seed,
    )
    doc_ids, preds, correct = [], [], []
    for item in bench:
        samples = adapter.extract_samples(item.doc, item.schema, K)
        cons = ground_predictions(consensus(samples), item.doc)
        report = score_fields(cons, item.golds)
        by_path = {p.path: p for p in cons}
        for fs in report.field_scores:
            p = by_path[fs.path]
            verb = verbalized_confidence(p)
            p = p.model_copy(update={"confidence": verb if verb is not None else 0.9})
            doc_ids.append(item.doc.doc_id)
            preds.append(p)
            correct.append(int(fs.correct))
    return np.array(doc_ids), preds, np.array(correct, dtype=float)


def evaluate(bench, name, rows):
    doc_ids, preds, correct = collect(bench)
    uniq = list(dict.fromkeys(doc_ids.tolist()))
    frac_grnd = float(np.mean([p.grounding is not None for p in preds]))

    # marginal guarantee: average coverage/risk over many random cal/test splits
    acc = {a: {"cp": [], "cg": [], "rg": []} for a in ALPHAS}
    for s in range(N_SPLITS):
        cal_docs, _ = split_calibration(uniq, 0.5, seed=SEED + s)
        cal_docs = set(cal_docs)
        cal = np.array([d in cal_docs for d in doc_ids])
        test = ~cal
        cal_preds = [p for p, m in zip(preds, cal) if m]
        test_preds = [p for p, m in zip(preds, test) if m]
        cal_conf = np.array([p.confidence for p in cal_preds])
        test_conf = np.array([p.confidence for p in test_preds])
        cal_y, test_y = correct[cal], correct[test]
        for alpha in ALPHAS:
            pooled = ConformalAbstention(alpha=alpha).fit(cal_conf.tolist(), cal_y.tolist())
            grouped = GroupConformalAbstention(alpha=alpha).fit(cal_preds, cal_y)
            pm, gm = pooled.accept(test_conf), grouped.accept(test_preds)
            acc[alpha]["cp"].append(float(pm.mean()))
            acc[alpha]["cg"].append(float(gm.mean()))
            if gm.any():
                acc[alpha]["rg"].append(float(1.0 - test_y[gm].mean()))

    for alpha in ALPHAS:
        cp, cg = float(np.mean(acc[alpha]["cp"])), float(np.mean(acc[alpha]["cg"]))
        rg = float(np.mean(acc[alpha]["rg"])) if acc[alpha]["rg"] else 0.0
        rows.append(
            {
                "dataset": name,
                "n_fields": len(preds),
                "frac_grnd": frac_grnd,
                "alpha": alpha,
                "cov_pooled": cp,
                "cov_grounded": cg,
                "cov_lift": cg - cp,
                "risk_grounded": rg,
                "held": rg <= alpha + 0.01,
            }
        )


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main() -> None:
    from benchmark.datasets import cord, funsd

    rows: list[dict] = []
    evaluate(cord.load(split="train", limit=400), "cord(real,n=400)", rows)
    evaluate(funsd.load(split="training") + funsd.load(split="testing"), "funsd(real,n=194)", rows)

    out = Path("paper/generated/grouped_conformal_real.md")
    cols = list(rows[0].keys())
    lines = [
        "# Grounding-conditioned vs pooled conformal on REAL documents",
        "",
        "CORD train + full FUNSD; realistic extractor over real text layers; accept score is",
        "verbalized (uninformative) confidence, grounding is the orthogonal signal.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lift = float(np.mean([r["cov_lift"] for r in rows]))
    held = all(r["held"] for r in rows)
    print(f"REAL-DATA grouped conformal: mean coverage lift +{lift:.3f}, guarantee held: {held}")
    for r in rows:
        print(
            f"  {r['dataset']:18s} a={r['alpha']:.2f}  pooled={r['cov_pooled']:.3f} "
            f"grounded={r['cov_grounded']:.3f}  lift=+{r['cov_lift']:.3f}  risk={r['risk_grounded']:.3f}"
        )


if __name__ == "__main__":
    main()
