#!/usr/bin/env python
"""Experiment A (repo issue #70): does PROVENANCE uniquely enable the Mondrian
conformal coverage lift, or do non-provenance covariates condition just as well?

Protocol reproduces scripts/phase2_method.py exactly (same seed=7, 40 document-
level 50/50 splits, logistic fusion fit on cal, add-one selective-risk
threshold), then runs an honest ablation over Mondrian taxonomies:

  provenance      : grounded | grounded x support-bin | support-bin
  non-provenance  : verbalized-bin | entailment-bin | consistency-bin |
                    field-type (cal-frequency AND fixed-rule variants) |
                    value-length bin | verbalized x entailment
  mixed           : support x verbalized (contains the provenance support signal)
  baseline        : pooled

Honest-evaluation contract:
  * document-level splits, fixed seeds (SPLIT_SEED=7, PERM_SEED=123)
  * all bin edges / field-type vocabularies computed on the CALIBRATION half
    of each split only (reference script used full-data quantiles; sanity gate
    below replicates that exact path to validate the harness)
  * held-at-nominal iff mean achieved test risk <= alpha (NO tolerance band)
  * violation fraction = fraction of the 40 splits with test risk > alpha
  * paired comparisons vs grounded x support: mean coverage difference across
    the same 40 splits + two-sided sign-flip permutation p-value

Usage:
  python run_ablation.py --data-dir <repo>/data --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

DATASETS = ["cord", "funsd", "xfund"]
ALPHAS = [0.05, 0.10]
N_SPLITS = 40
SPLIT_SEED = 7      # same as scripts/phase2_method.py
PERM_SEED = 123     # sign-flip permutation test
N_PERM = 20000
SIGNAL_KEYS = ["verbalized", "consistency", "support", "grounded", "entailment"]

PROVENANCE_TAX = {"grounded", "grounded x support", "support-bin"}
MIXED_TAX = {"support x verb"}
BASELINE_TAX = {"pooled"}
# everything else is non-provenance


# ----------------------------------------------------------------- conformal
def conformal_threshold(score: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Smallest threshold whose add-one empirical selective risk on cal <= alpha.
    Identical to scripts/phase2_method.py."""
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
    """Per-group thresholds fit on cal; accept mask on test.
    Identical to scripts/phase2_method.py (unseen/empty groups -> reject)."""
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


# ---------------------------------------------------------------- taxonomies
def cal_bins(x: np.ndarray, cal: np.ndarray, q) -> np.ndarray:
    """Quantile bins with edges computed on the calibration half ONLY."""
    edges = np.unique(np.quantile(x[cal], q))
    if len(edges) == 0:
        return np.zeros(len(x), int)
    return np.digitize(x, edges)


def leaf_key(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path).split(".")[-1].lower()


def field_type_rule(path: str) -> int:
    """Fixed keyword rule (no data dependence at all -> trivially leakage-free).
    Priority: date > contact > amount > count > name > other."""
    lf = leaf_key(path)
    if any(k in lf for k in ("date", "datum")):
        return 1
    if any(k in lf for k in ("phone", "fax", "tel", "mail")):
        return 2
    if any(k in lf for k in ("price", "total", "amount", "betrag", "cash",
                             "credit", "change", "discount", "tax", "service")):
        return 3
    if any(k in lf for k in ("cnt", "qty", "count", "no_of", "number", "plz")):
        return 4
    if any(k in lf for k in ("nm", "name", "company", "from", "to", "subject")):
        return 5
    return 0


def make_taxonomies(sig, values, paths, cal: np.ndarray) -> dict[str, np.ndarray]:
    """All group arrays for one split. Everything data-dependent uses cal only."""
    n = len(cal)
    g = sig["grounded"].astype(int)
    sb = cal_bins(sig["support"], cal, (0.34, 0.67))          # 3 bins (as original grid)
    vb = cal_bins(sig["verbalized"], cal, (0.5,))             # 2 bins
    eb = cal_bins(sig["entailment"], cal, (0.5,))             # 2 bins
    cb = cal_bins(sig["consistency"], cal, (0.5,))            # 2 bins

    # field-type by cal frequency: leaf keys with >=25 cal fields get own group
    lks = np.array([leaf_key(p) for p in paths])
    cal_keys, cal_counts = np.unique(lks[cal], return_counts=True)
    vocab = {k: i + 1 for i, k in enumerate(sorted(cal_keys[cal_counts >= 25]))}
    ft_freq = np.array([vocab.get(k, 0) for k in lks], int)

    ft_rule = np.array([field_type_rule(p) for p in paths], int)

    vlen = np.array(
        [0 if (v is None or len(str(v)) <= 2) else 1 for v in values], int
    )  # short (<=2 chars) vs longer

    return {
        "pooled": np.zeros(n, int),
        # provenance
        "grounded": g,
        "grounded x support": g * 3 + sb,
        "support-bin": sb,
        # non-provenance
        "verbalized-bin": vb,
        "entailment-bin": eb,
        "consistency-bin": cb,
        "fieldtype-freq": ft_freq,
        "fieldtype-rule": ft_rule,
        "value-length": vlen,
        "verb x entail": vb * 2 + eb,
        # mixed (listed with non-provenance in the ablation spec, but contains
        # the provenance support signal -- flagged separately in the report)
        "support x verb": sb * 2 + vb,
    }


# -------------------------------------------------------------- sanity gate
def sanity_gate(data_dir: Path) -> dict:
    """Reproduce scripts/phase2_method.py EXACTLY on CORD (full-data quantile
    bins, seed 7, 40 splits) and check the two published numbers at alpha=0.10:
      fusion/pooled            cov ~ 0.056, risk ~ 0.105
      fusion/grounded x support cov ~ 0.114, risk ~ 0.122
    """
    recs = json.loads((data_dir / "apivlm_perfield_rich_cord.json").read_text())
    docs = np.array([r["doc_id"] for r in recs])
    y = np.array([r["correct"] for r in recs], float)
    X = np.column_stack(
        [np.array([float(r.get(k, 0.0)) for r in recs]) for k in SIGNAL_KEYS]
    )
    sig = {k: X[:, i] for i, k in enumerate(SIGNAL_KEYS)}

    def full_bins(x, q=(0.5,)):  # reference used full-data quantiles
        return np.digitize(x, np.quantile(x, q))

    taxes = {
        "pooled": np.zeros(len(y), int),
        "grounded x support": sig["grounded"].astype(int) * 3
        + full_bins(sig["support"], (0.34, 0.67)),
    }
    uniq = list(dict.fromkeys(docs.tolist()))
    rng = np.random.default_rng(SPLIT_SEED)
    out = {t: {"cov": [], "risk": []} for t in taxes}
    for _ in range(N_SPLITS):
        perm = rng.permutation(uniq)
        cal_docs = set(perm[: len(uniq) // 2].tolist())
        cal = np.array([d in cal_docs for d in docs])
        test = ~cal
        lr = LogisticRegression(max_iter=1000).fit(X[cal], y[cal])
        fusion = lr.predict_proba(X)[:, 1]
        for t, gr in taxes.items():
            acc = apply_grouped(fusion, gr, y, cal, test, 0.10)
            yt = y[test]
            out[t]["cov"].append(float(acc.mean()))
            out[t]["risk"].append(float(1.0 - yt[acc].mean()) if acc.any() else 0.0)

    got = {
        t: {"cov": float(np.mean(d["cov"])), "risk": float(np.mean(d["risk"]))}
        for t, d in out.items()
    }
    expect = {
        "pooled": {"cov": 0.056, "risk": 0.105},
        "grounded x support": {"cov": 0.114, "risk": 0.122},
    }
    ok = all(
        abs(got[t][m] - expect[t][m]) <= 0.005 for t in expect for m in ("cov", "risk")
    )
    return {"pass": ok, "expected": expect, "got": got, "tolerance": 0.005}


# ------------------------------------------------------------------ ablation
def run_dataset(recs: list[dict]) -> dict:
    docs = np.array([r["doc_id"] for r in recs])
    y = np.array([r["correct"] for r in recs], float)
    values = [r.get("value") for r in recs]
    paths = [r["path"] for r in recs]
    X = np.column_stack(
        [np.array([float(r.get(k, 0.0)) for r in recs]) for k in SIGNAL_KEYS]
    )
    sig = {k: X[:, i] for i, k in enumerate(SIGNAL_KEYS)}
    uniq = list(dict.fromkeys(docs.tolist()))
    rng = np.random.default_rng(SPLIT_SEED)

    per_split: dict = {}  # (score, tax, alpha) -> {"cov": [...], "risk": [...]}
    for _sp in range(N_SPLITS):
        perm = rng.permutation(uniq)
        cal_docs = set(perm[: len(uniq) // 2].tolist())
        cal = np.array([d in cal_docs for d in docs])
        test = ~cal
        yt = y[test]

        lr = LogisticRegression(max_iter=1000).fit(X[cal], y[cal])
        fusion = lr.predict_proba(X)[:, 1]
        scores = {"fusion": fusion, "verbalized": sig["verbalized"]}

        taxes = make_taxonomies(sig, values, paths, cal)  # cal-only edges
        for sname, sc in scores.items():
            for tname, gr in taxes.items():
                for a in ALPHAS:
                    acc = apply_grouped(sc, gr, y, cal, test, a)
                    cov = float(acc.mean())
                    risk = float(1.0 - yt[acc].mean()) if acc.any() else 0.0
                    d = per_split.setdefault((sname, tname, a), {"cov": [], "risk": []})
                    d["cov"].append(cov)
                    d["risk"].append(risk)

    # aggregate under the strict contract
    grid: dict = {}
    for (sname, tname, a), d in per_split.items():
        cov = np.array(d["cov"])
        risk = np.array(d["risk"])
        grid.setdefault(str(a), {}).setdefault(sname, {})[tname] = {
            "coverage_mean": round(float(cov.mean()), 4),
            "coverage_std": round(float(cov.std(ddof=1)), 4),
            "risk_mean": round(float(risk.mean()), 4),
            "violation_frac": round(float((risk > a).mean()), 4),
            "held": bool(risk.mean() <= a),  # NOMINAL alpha, no tolerance
            "category": (
                "baseline" if tname in BASELINE_TAX
                else "provenance" if tname in PROVENANCE_TAX
                else "mixed" if tname in MIXED_TAX
                else "non-provenance"
            ),
            "cov_per_split": [round(c, 5) for c in d["cov"]],
            "risk_per_split": [round(r, 5) for r in d["risk"]],
        }
    return grid


def signflip_p(diff: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p-value for mean(diff) = 0."""
    obs = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
    null = np.abs((flips * diff).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + N_PERM))


def paired_tests(grid: dict) -> dict:
    """Every taxonomy vs grounded x support, paired over the same 40 splits."""
    out: dict = {}
    rng = np.random.default_rng(PERM_SEED)
    for a, by_score in grid.items():
        for sname, by_tax in by_score.items():
            ref = np.array(by_tax["grounded x support"]["cov_per_split"])
            for tname, cell in by_tax.items():
                if tname == "grounded x support":
                    continue
                diff = np.array(cell["cov_per_split"]) - ref
                out.setdefault(a, {}).setdefault(sname, {})[tname] = {
                    "mean_cov_diff_vs_gxs": round(float(diff.mean()), 4),
                    "signflip_p": round(signflip_p(diff, rng), 5),
                }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        default="/Users/challagullayoshitha/Documents/Research/verifydoc/data",
    )
    ap.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    print("== sanity gate: reproduce phase2_method.py on CORD (alpha=0.10) ==")
    gate = sanity_gate(data_dir)
    for t, v in gate["got"].items():
        e = gate["expected"][t]
        print(f"  {t:20} cov {v['cov']:.4f} (exp {e['cov']}) "
              f"risk {v['risk']:.4f} (exp {e['risk']})")
    if not gate["pass"]:
        print("SANITY GATE FAILED -- aborting, harness does not reproduce.")
        sys.exit(1)
    print("  PASS (within ±0.005)\n")

    results = {
        "seeds": {"split_seed": SPLIT_SEED, "perm_seed": PERM_SEED,
                  "n_splits": N_SPLITS, "n_perm": N_PERM},
        "protocol": "40 doc-level 50/50 splits; fusion=LR on cal over "
                    f"{SIGNAL_KEYS}; add-one selective-risk threshold; "
                    "Mondrian per group; cal-only bin edges; held iff mean "
                    "test risk <= alpha (no tolerance).",
        "sanity_gate": gate,
        "grid": {},
        "paired_vs_grounded_x_support": {},
    }
    for ds in DATASETS:
        recs = json.loads((data_dir / f"apivlm_perfield_rich_{ds}.json").read_text())
        print(f"== {ds}: n={len(recs)} fields, "
              f"{len(set(r['doc_id'] for r in recs))} docs ==")
        grid = run_dataset(recs)
        results["grid"][ds] = grid
        results["paired_vs_grounded_x_support"][ds] = paired_tests(grid)
        for a in map(str, ALPHAS):
            for sname in ("fusion", "verbalized"):
                cells = grid[a][sname]
                order = sorted(cells.items(),
                               key=lambda kv: -kv[1]["coverage_mean"])
                print(f"  alpha={a} score={sname}")
                for tname, c in order:
                    print(f"    {tname:20} [{c['category'][:4]}] "
                          f"cov {c['coverage_mean']:.4f}±{c['coverage_std']:.4f} "
                          f"risk {c['risk_mean']:.4f} viol {c['violation_frac']:.2f} "
                          f"held={c['held']}")
        print()

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
