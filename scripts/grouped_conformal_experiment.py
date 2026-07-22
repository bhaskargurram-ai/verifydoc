#!/usr/bin/env python
"""Controlled study of grounding-conditioned (Mondrian) conformal risk control.

Isolates the mechanism in the regime we actually measured on the frontier VLM
(paper/generated/cord-apivlm): the model's confidence score is inflated and
near-uninformative for ranking errors, while *grounding* strongly predicts
correctness. In that regime a single pooled conformal threshold must abstain
broadly, whereas conditioning the threshold on provenance accepts the reliable
well-grounded fields --- recovering coverage at the same risk guarantee.

We construct a field population directly (deterministic, seeded) so the effect
is not confounded by extractor or OCR-formatting artifacts:
  - a fraction ``p_grounded`` of fields are grounded, with accuracy
    ``acc_grounded`` (high); the rest are ungrounded, with accuracy
    ``acc_ungrounded`` (low);
  - the accept score (verbalized confidence) is uniform noise, uncorrelated
    with correctness --- as observed for the real VLM.
Reports, for several conditions, pooled vs grounding-conditioned coverage and
the achieved (guaranteed) selective risk. Writes
paper/generated/grouped_conformal.{md}.

Usage: python scripts/grouped_conformal_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.calibration import ConformalAbstention, GroupConformalAbstention  # noqa: E402
from verifydoc.types import FieldPrediction, Grounding  # noqa: E402

ALPHA = 0.05
N = 1200
N_TRIALS = 200
CONDITIONS = [
    # (p_grounded, acc_grounded, acc_ungrounded)
    (0.60, 0.97, 0.55),
    (0.50, 0.95, 0.40),
    (0.40, 0.98, 0.50),
    (0.70, 0.96, 0.60),
]


def _make_population(rng, p_grounded, acc_g, acc_u):
    preds, correct = [], []
    for _ in range(N):
        grounded = rng.random() < p_grounded
        acc = acc_g if grounded else acc_u
        y = int(rng.random() < acc)
        conf = float(rng.random())  # uninformative accept score (uncorrelated with y)
        support = 0.9 if grounded else 0.1
        preds.append(
            FieldPrediction(
                path="f", value="v", confidence=conf, grounding=Grounding(page=0, support=support)
            )
        )
        correct.append(y)
    return preds, np.array(correct, dtype=float)


def _run_condition(p_grounded, acc_g, acc_u):
    rng = np.random.default_rng(7)
    pooled_cov, grouped_cov, pooled_risk, grouped_risk = [], [], [], []
    for _ in range(N_TRIALS):
        cal_p, cal_y = _make_population(rng, p_grounded, acc_g, acc_u)
        test_p, test_y = _make_population(rng, p_grounded, acc_g, acc_u)
        cal_conf = np.array([p.confidence for p in cal_p])
        test_conf = np.array([p.confidence for p in test_p])

        pooled = ConformalAbstention(alpha=ALPHA).fit(cal_conf, cal_y)
        grouped = GroupConformalAbstention(alpha=ALPHA).fit(cal_p, cal_y)
        pm, gm = pooled.accept(test_conf), grouped.accept(test_p)
        pooled_cov.append(pm.mean())
        grouped_cov.append(gm.mean())
        if pm.any():
            pooled_risk.append(1.0 - test_y[pm].mean())
        if gm.any():
            grouped_risk.append(1.0 - test_y[gm].mean())
    return {
        "p_grnd": p_grounded,
        "acc_grnd": acc_g,
        "acc_ungrnd": acc_u,
        "cov_pooled": float(np.mean(pooled_cov)),
        "cov_grounded": float(np.mean(grouped_cov)),
        "cov_lift": float(np.mean(grouped_cov) - np.mean(pooled_cov)),
        "risk_grounded": float(np.mean(grouped_risk)) if grouped_risk else 0.0,
        "held": (float(np.mean(grouped_risk)) if grouped_risk else 0.0) <= ALPHA + 0.01,
    }


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main() -> None:
    rows = [_run_condition(*c) for c in CONDITIONS]
    cols = list(rows[0].keys())
    out = Path("paper/generated/grouped_conformal.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Grounding-conditioned vs pooled conformal (controlled study, alpha={ALPHA})",
        "",
        "Uninformative accept score + provenance-predictive correctness (the real-VLM regime).",
        "Coverage is the fraction of fields auto-accepted; risk_grounded is the achieved",
        "selective risk of the grounding-conditioned policy (must stay <= alpha).",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    lift = float(np.mean([r["cov_lift"] for r in rows]))
    held = all(r["held"] for r in rows)
    print(f"mean coverage lift +{lift:.3f}, guarantee held every condition: {held}")
    for r in rows:
        print(
            f"  p_grnd={r['p_grnd']:.2f} acc={r['acc_grnd']:.2f}/{r['acc_ungrnd']:.2f}  "
            f"pooled={r['cov_pooled']:.3f} grounded={r['cov_grounded']:.3f}  "
            f"lift=+{r['cov_lift']:.3f}  risk={r['risk_grounded']:.3f}"
        )


if __name__ == "__main__":
    main()
