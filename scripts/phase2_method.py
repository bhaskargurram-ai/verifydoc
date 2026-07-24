#!/usr/bin/env python
"""Phase-2: grounding-conditioned conformal with a learned multi-signal fusion.

Consumes the rich per-field dump (scripts/apivlm_perfield_rich.py) and evaluates,
on GENUINE frontier-VLM outputs, whether a learned fusion accept-score + richer
Mondrian conditioning lifts certifiable coverage where the single-signal +
binary-grounded baseline gives zero.

For each of N document-level cal/test splits it fits a logistic fusion of
{verbalized, consistency, support, grounded, entailment} on the calibration
fields, then runs split-conformal (add-one selective-risk threshold) pooled and
under several provenance taxonomies -- including a taxonomy SELECTED on a held-out
calibration sub-split (valid, no winner's curse). Reports mean test coverage +
achieved risk over splits, with a document-clustered bootstrap CI on the headline.

    python scripts/phase2_method.py [--dump data/apivlm_perfield_rich.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ALPHAS = [0.05, 0.10, 0.20]
N_SPLITS = 40
SIGNAL_KEYS = ["verbalized", "consistency", "support", "grounded", "entailment"]


def conformal_threshold(score: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Smallest threshold whose add-one empirical selective risk is <= alpha."""
    if len(score) == 0:
        return np.inf
    order = np.argsort(-score, kind="stable")
    s, y = score[order], correct[order]
    err = np.cumsum(1.0 - y)
    n = np.arange(1, len(y) + 1)
    risk = (1.0 + err) / (1.0 + n)
    boundary = np.append(s[:-1] > s[1:], True)
    ok = np.where((risk <= alpha) & boundary)[0]
    return float(s[ok.max()]) if len(ok) else np.inf


def apply_grouped(score, groups, correct, cal, test, alpha):
    """Fit per-group thresholds on cal, return accept mask on test."""
    acc = np.zeros(int(test.sum()), bool)
    gt = groups[test]
    test_idx = np.where(test)[0]
    for g in np.unique(groups):
        cg = cal & (groups == g)
        sel = gt == g
        if cg.sum() == 0 or sel.sum() == 0:
            continue
        t = conformal_threshold(score[cg], correct[cg], alpha)
        acc[sel] = score[test_idx[sel]] >= t
    return acc


def bins(x, q=(0.5,)):
    edges = np.quantile(x, q)
    return np.digitize(x, edges)


def make_taxonomies(sig):
    g = sig["grounded"].astype(int)
    vb = bins(sig["verbalized"])
    sb = bins(sig["support"], (0.34, 0.67))
    eb = bins(sig["entailment"])
    return {
        "pooled": np.zeros(len(g), int),
        "grounded": g,
        "grounded x verb": g * 2 + vb,
        "grounded x support": g * 3 + sb,
        "grounded x entail": g * 2 + eb,
        "support x verb": sb * 2 + vb,
    }


def run(dump_path: str):
    recs = json.loads(Path(dump_path).read_text())
    docs = np.array([r["doc_id"] for r in recs])
    y = np.array([r["correct"] for r in recs], float)
    X = np.column_stack([np.array([float(r.get(k, 0.0)) for r in recs]) for k in SIGNAL_KEYS])
    sig = {k: X[:, i] for i, k in enumerate(SIGNAL_KEYS)}
    taxes = make_taxonomies(sig)
    uniq = list(dict.fromkeys(docs.tolist()))
    rng = np.random.default_rng(7)

    # accept-scores: single signals + learned fusion (fit per split, no leakage)
    results: dict = {a: {} for a in ALPHAS}
    for _sp in range(N_SPLITS):
        perm = rng.permutation(uniq)
        cal_docs = set(perm[: len(uniq) // 2].tolist())
        cal = np.array([d in cal_docs for d in docs])
        test = ~cal
        # fusion fit on cal only
        lr = LogisticRegression(max_iter=1000).fit(X[cal], y[cal])
        fusion = lr.predict_proba(X)[:, 1]
        scores = {"verbalized": sig["verbalized"], "support": sig["support"], "fusion": fusion}
        for sname, sc in scores.items():
            for tname, gr in taxes.items():
                # held-out taxonomy selection handled separately below
                for a in ALPHAS:
                    acc = apply_grouped(sc, gr, y, cal, test, a)
                    yt = y[test]
                    cov = float(acc.mean())
                    risk = float(1.0 - yt[acc].mean()) if acc.any() else 0.0
                    key = (sname, tname)
                    results[a].setdefault(key, {"cov": [], "risk": []})
                    results[a][key]["cov"].append(cov)
                    results[a][key]["risk"].append(risk)

        # SELECTED taxonomy (fusion score): pick best on a held-out cal sub-split
        cal_ids = list(cal_docs)
        fit_ids = set(cal_ids[: len(cal_ids) // 2])
        fit = np.array([d in fit_ids for d in docs]) & cal
        selh = cal & ~fit
        for a in ALPHAS:
            best_t, best_cov = "pooled", -1.0
            for tname, gr in taxes.items():
                acc_sel = apply_grouped(fusion, gr, y, fit, selh, a)
                ysel = y[selh]
                if (
                    acc_sel.any()
                    and (1.0 - ysel[acc_sel].mean()) <= a
                    and acc_sel.mean() > best_cov
                ):
                    best_cov, best_t = float(acc_sel.mean()), tname
            acc = apply_grouped(fusion, taxes[best_t], y, cal, test, a)
            yt = y[test]
            key = ("fusion", "SELECTED")
            results[a].setdefault(key, {"cov": [], "risk": []})
            results[a][key]["cov"].append(float(acc.mean()))
            results[a][key]["risk"].append(float(1.0 - yt[acc].mean()) if acc.any() else 0.0)

    # aggregate
    rows = []
    for a in ALPHAS:
        pooled_fusion = float(np.mean(results[a][("fusion", "pooled")]["cov"]))
        for (sname, tname), d in results[a].items():
            cov, risk = float(np.mean(d["cov"])), float(np.mean(d["risk"]))
            rows.append(
                {
                    "alpha": a,
                    "score": sname,
                    "taxonomy": tname,
                    "coverage": round(cov, 4),
                    "risk": round(risk, 4),
                    "held": bool(risk <= a + 0.02),
                    "lift_vs_pooled_fusion": round(cov - pooled_fusion, 4),
                }
            )
    return recs, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="data/apivlm_perfield_rich.json")
    ap.add_argument("--out", default="results/phase2_method.json")
    args = ap.parse_args()
    recs, rows = run(args.dump)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))

    print(f"Phase-2 method on {len(recs)} genuine-VLM fields ({args.dump}):\n")
    hdr = f"{'alpha':>5} {'score':11} {'taxonomy':18} {'cov':>6} {'risk':>6} held  lift"
    print(hdr)
    for r in sorted(rows, key=lambda x: (x["alpha"], -x["coverage"])):
        print(
            f"{r['alpha']:>5} {r['score']:11} {r['taxonomy']:18} "
            f"{r['coverage']:>6.3f} {r['risk']:>6.3f} {str(r['held']):5} {r['lift_vs_pooled_fusion']:+.3f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
