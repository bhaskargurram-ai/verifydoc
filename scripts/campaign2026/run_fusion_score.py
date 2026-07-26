#!/usr/bin/env python
"""EXP-F: definitive score-side table — scale EXP-C's hgb_split recipe to the
NEW 2x dumps (cord=13,859 fields/800 docs; funsd=1,999/175; xfund=523/42).

Protocol (honest-evaluation contract, identical stream to EXP-C /
phase2_method.py): seed=7, 40 document-level 50/50 cal/test splits; add-one
selective-risk conformal threshold; NOMINAL alpha (held iff mean test risk <=
alpha, no tolerance band); risk 0-filled when nothing accepted.

SPLIT protocol (the EXP-C validity fix, used for every learned score here):
within the calibration half, model + all data-dependent feature transforms
(leaf-key vocab) are fit on the doc-level FIT-half only; the conformal
threshold AND every Mondrian taxonomy's data-dependent part (support-bin
edges, fieldtype-freq vocab) are fit on the untouched VAL-half only; test is
never touched. baseline_lr (the control) additionally runs under the
reference cal-protocol (model + threshold on full cal) for parity with
phase2_method.py — EXP-C showed the low-capacity LR barely contaminates.

Scores (grid rows):
  baseline_lr            LR on the 5 raw signals, cal-protocol (control)
  baseline_lr_split      same LR, split-protocol threshold (honest control)
  hgb_split              depth-3 HistGradientBoosting on raw signals +
                         engineered features (log value length, digit frac,
                         is-numeric, support==0, cons*verb, ent*grd,
                         fixed-rule fieldtype one-hots, fit-frequency
                         leaf-key one-hots), doc-level early stopping
  hgb_split_noent        drop {entailment, ent_x_grd}   (NLI-stage ablation)
  hgb_split_nofieldtype  drop {ftr_*, lk_*}             (fieldtype ablation)
# DECISION: "fieldtype features" = BOTH the fixed-rule one-hots (ftr_*) AND
# the leaf-key frequency one-hots (lk_*). The subsumption question compares
# score-side fieldtype against fieldtype-freq/-rule Mondrian taxonomies,
# which are built from exactly these two constructions (exp_a), so the
# ablation must remove both to move the covariate cleanly out of the score.

Taxonomies (Mondrian at the threshold step; data-dependent parts from the
threshold-fitting rows ONLY — val-half for *_split, cal for baseline_lr):
  pooled | support-bin (quantile edges 0.34/0.67) |
  fieldtype-freq (leaf keys with >=25 threshold-row fields get own group;
  exp_a definition) | fieldtype-rule (fixed keyword rule, data-free)

Alphas: 0.05, 0.10. Reported per cell: coverage mean+-std over 40 splits,
mean risk, violation fraction, held (nominal).

SANITY GATE (abort on failure): on the OLD 6.9k CORD dump, reproduce EXP-C
within +-0.01 (cov AND risk) at alpha=0.10:
  hgb_split pooled 0.2660/0.0921 ; hgb_split support-bin 0.2754/0.0993 ;
  baseline_lr pooled 0.0559/0.1055.

Stats: paired per-split coverage diffs with two-sided sign-flip permutation
p-values (20k perms, seed 123) vs baseline_lr pooled, plus the subsumption
contrasts (fieldtype-Mondrian on/off x fieldtype score-side on/off).

Usage:
  python run_final_score.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).parent
OLD_DATA = Path("archive/pre-campaign-2026-07-26/data")
NEW_DATA = Path("data")

DATASETS = ["cord", "funsd", "xfund"]
ALPHAS = [0.05, 0.10]
N_SPLITS = 40
SPLIT_SEED = 7        # identical split stream to phase2_method.py / EXP-A / EXP-C
SUB_SEED = 9000       # + split idx -> doc-level cal sub-split (fit/val)
INNER_SEED = 59000    # + split idx -> doc-level inner fit sub-split (hgb stop)
PERM_SEED = 123       # sign-flip permutation test
N_PERM = 20000
SIGNAL_KEYS = ["verbalized", "consistency", "support", "entailment", "grounded"]
VOCAB_MIN_COUNT = 25  # leaf keys need >= this many fit fields for own one-hot
VOCAB_MAX = 10
FT_FREQ_MIN = 25      # exp_a fieldtype-freq taxonomy vocab threshold
HGB_MAX_ITER = 300
HGB_STEP = 10
HGB_PATIENCE = 5

TAXONOMIES = ["pooled", "support-bin", "fieldtype-freq", "fieldtype-rule"]
NOENT = {"entailment", "ent_x_grd"}

# EXP-C targets on OLD cord (from exp_c/results.json), tolerance +-0.01
SANITY_TARGETS = {
    ("baseline_lr", "pooled"): (0.0559, 0.1055),
    ("hgb_split", "pooled"): (0.2660, 0.0921),
    ("hgb_split", "support-bin"): (0.2754, 0.0993),
}
SANITY_TOL = 0.01


# ----------------------------------------------------------------- conformal
def conformal_threshold(score: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Smallest threshold whose add-one empirical selective risk on the
    threshold-fitting rows <= alpha. Verbatim from scripts/phase2_method.py."""
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


def apply_grouped(score, groups, correct, thr_mask, test, alpha):
    """Per-group thresholds fit on thr_mask rows; accept mask on test.
    Groups unseen in thr_mask reject everything (acc stays False)."""
    acc = np.zeros(int(test.sum()), bool)
    gt = groups[test]
    test_idx = np.where(test)[0]
    for g in np.unique(groups):
        cg = thr_mask & (groups == g)
        sel = gt == g
        if cg.sum() == 0 or sel.sum() == 0:
            continue
        t = conformal_threshold(score[cg], correct[cg], alpha)
        acc[sel] = score[test_idx[sel]] >= t
    return acc


def cal_bins(x: np.ndarray, mask: np.ndarray, q) -> np.ndarray:
    """Quantile bins with edges computed on the threshold-fitting rows ONLY."""
    edges = np.unique(np.quantile(x[mask], q))
    if len(edges) == 0:
        return np.zeros(len(x), int)
    return np.digitize(x, edges)


# ------------------------------------------------------------------ features
def leaf_key(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path).split(".")[-1].lower()


def field_type_rule(path: str) -> int:
    """Fixed keyword rule (no data dependence). Verbatim from exp_a/exp_c."""
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


def static_features(recs: list[dict]):
    """Cal-independent features: pure per-record functions (leakage-free)."""
    vals = [str(r.get("value")) if r.get("value") is not None else "" for r in recs]
    verb = np.array([float(r.get("verbalized", 0.0)) for r in recs])
    cons = np.array([float(r.get("consistency", 0.0)) for r in recs])
    supp = np.array([float(r.get("support", 0.0)) for r in recs])
    ent = np.array([float(r.get("entailment", 0.0)) for r in recs])
    grd = np.array([float(r.get("grounded", 0.0)) for r in recs])
    log_len = np.log1p(np.array([len(v) for v in vals], float))
    digit_frac = np.array(
        [sum(c.isdigit() for c in v) / len(v) if v else 0.0 for v in vals]
    )
    is_numeric = np.array(
        [1.0 if v.strip() and re.fullmatch(r"[-+]?[\d.,]+", v.strip()) else 0.0
         for v in vals]
    )
    cols = {
        "verbalized": verb,
        "consistency": cons,
        "support": supp,
        "entailment": ent,
        "grounded": grd,
        "log_len": log_len,
        "digit_frac": digit_frac,
        "is_numeric": is_numeric,
        "support_zero": (supp == 0.0).astype(float),
        "cons_x_verb": cons * verb,
        "ent_x_grd": ent * grd,
    }
    rules = np.array([field_type_rule(r["path"]) for r in recs])
    for k in range(6):
        cols[f"ftr_{k}"] = (rules == k).astype(float)
    return cols


def split_matrix(static_cols, leaf_keys, fit_mask):
    """Engineered matrix; the data-dependent leaf-key one-hot vocabulary
    (>= VOCAB_MIN_COUNT fit fields, top VOCAB_MAX by count) is built from
    fit_mask rows ONLY. Verbatim from exp_c."""
    names = list(static_cols.keys())
    mats = [static_cols[n] for n in names]
    keys, counts = np.unique(leaf_keys[fit_mask], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    vocab = [k for k in keys[order][counts[order] >= VOCAB_MIN_COUNT][:VOCAB_MAX]]
    for k in sorted(vocab):
        names.append(f"lk_{k}")
        mats.append((leaf_keys == k).astype(float))
    return np.column_stack(mats), names


def drop_cols(X, names, drop):
    keep = [j for j, n in enumerate(names) if n not in drop]
    return X[:, keep]


def fieldtype_cols(names):
    return {n for n in names if n.startswith("ftr_") or n.startswith("lk_")}


# -------------------------------------------------------------------- models
def fit_hgb(X, y, fit_mask, val_mask, refit_mask):
    """Doc-level early stopping (warm_start on fit_mask, val log-loss on
    val_mask, patience), refit at best n_iter on refit_mask. Verbatim exp_c."""
    m = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, max_iter=HGB_STEP,
        warm_start=True, early_stopping=False, random_state=0,
    )
    best_ll, best_iter, bad = np.inf, HGB_STEP, 0
    for it in range(HGB_STEP, HGB_MAX_ITER + 1, HGB_STEP):
        m.set_params(max_iter=it)
        m.fit(X[fit_mask], y[fit_mask])
        ll = log_loss(y[val_mask], m.predict_proba(X[val_mask])[:, 1],
                      labels=[0.0, 1.0])
        if ll < best_ll - 1e-6:
            best_ll, best_iter, bad = ll, it, 0
        else:
            bad += 1
            if bad >= HGB_PATIENCE:
                break
    final = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.1, max_iter=best_iter,
        early_stopping=False, random_state=0,
    ).fit(X[refit_mask], y[refit_mask])
    return final, best_iter


# --------------------------------------------------------------- taxonomies
def make_taxonomies(supp, lks, ftr, thr_mask):
    """The 4 grid taxonomies; every data-dependent part uses thr_mask rows
    ONLY (the rows the conformal threshold is fit on). fieldtype-freq is the
    exp_a definition: leaf keys with >= FT_FREQ_MIN thr fields get their own
    group (sorted order), everything else group 0."""
    keys, counts = np.unique(lks[thr_mask], return_counts=True)
    vocab = {k: i + 1 for i, k in enumerate(sorted(keys[counts >= FT_FREQ_MIN]))}
    ft_freq = np.array([vocab.get(k, 0) for k in lks], int)
    return {
        "pooled": np.zeros(len(lks), int),
        "support-bin": cal_bins(supp, thr_mask, (0.34, 0.67)),
        "fieldtype-freq": ft_freq,
        "fieldtype-rule": ftr,
    }


# ------------------------------------------------------------------ main run
def run_dataset(recs: list[dict], taxonomies=TAXONOMIES,
                scores_to_run=("baseline_lr", "baseline_lr_split", "hgb_split",
                               "hgb_split_noent", "hgb_split_nofieldtype"),
                verbose=True) -> dict:
    docs = np.array([r["doc_id"] for r in recs])
    y = np.array([r["correct"] for r in recs], float)
    supp = np.array([float(r.get("support", 0.0)) for r in recs])
    Xb = np.column_stack(
        [np.array([float(r.get(k, 0.0)) for r in recs]) for k in SIGNAL_KEYS]
    )
    static_cols = static_features(recs)
    lks = np.array([leaf_key(r["path"]) for r in recs])
    ftr = np.array([field_type_rule(r["path"]) for r in recs], int)
    uniq = list(dict.fromkeys(docs.tolist()))
    rng = np.random.default_rng(SPLIT_SEED)

    score_metrics: dict = {}
    hgb_iters: list[int] = []

    def record(name, sc, thr_mask, test, yt, taxes):
        d = score_metrics.setdefault(name, {"auroc": [], "cells": {}})
        d["auroc"].append(float(roc_auc_score(yt, sc[test])))
        for a in ALPHAS:
            for tname in taxonomies:
                acc = apply_grouped(sc, taxes[tname], y, thr_mask, test, a)
                cell = d["cells"].setdefault((a, tname), {"cov": [], "risk": []})
                cell["cov"].append(float(acc.mean()))
                cell["risk"].append(float(1.0 - yt[acc].mean()) if acc.any() else 0.0)

    t0 = time.time()
    for sp in range(N_SPLITS):
        perm = rng.permutation(uniq)
        cal_docs = set(perm[: len(uniq) // 2].tolist())
        cal = np.array([d in cal_docs for d in docs])
        test = ~cal
        yt = y[test]

        # doc-level fit/val sub-split within cal (split protocol)
        rng_sub = np.random.default_rng(SUB_SEED + sp)
        cal_list = [d for d in uniq if d in cal_docs]
        sub_perm = rng_sub.permutation(cal_list)
        fit_docs = set(sub_perm[: len(cal_list) // 2].tolist())
        fit = np.array([d in fit_docs for d in docs]) & cal
        val = cal & ~fit

        # inner doc-level sub-split of FIT (hgb early stopping; val untouched)
        rng_in = np.random.default_rng(INNER_SEED + sp)
        fit_list = [d for d in cal_list if d in fit_docs]
        in_perm = rng_in.permutation(fit_list)
        n_stop = max(1, len(fit_list) // 4)
        stop_docs = set(in_perm[:n_stop].tolist())
        f_stop = np.array([d in stop_docs for d in docs]) & fit
        f_core = fit & ~f_stop

        taxes_cal = make_taxonomies(supp, lks, ftr, cal)   # cal-protocol rows
        taxes_val = make_taxonomies(supp, lks, ftr, val)   # split-protocol rows

        # --- control: baseline LR, cal-protocol (reference parity)
        if "baseline_lr" in scores_to_run:
            lr_b = LogisticRegression(max_iter=1000).fit(Xb[cal], y[cal])
            record("baseline_lr", lr_b.predict_proba(Xb)[:, 1],
                   cal, test, yt, taxes_cal)

        # --- honest control: baseline LR under the split protocol
        if "baseline_lr_split" in scores_to_run:
            lr_bs = LogisticRegression(max_iter=1000).fit(Xb[fit], y[fit])
            record("baseline_lr_split", lr_bs.predict_proba(Xb)[:, 1],
                   val, test, yt, taxes_val)

        # --- split-protocol HGB variants (transforms + model from fit only)
        Xf, names_f = split_matrix(static_cols, lks, fit)
        if "hgb_split" in scores_to_run:
            hgb_f, best_iter = fit_hgb(Xf, y, f_core, f_stop, fit)
            hgb_iters.append(best_iter)
            record("hgb_split", hgb_f.predict_proba(Xf)[:, 1],
                   val, test, yt, taxes_val)

        if "hgb_split_noent" in scores_to_run:
            Xf_ne = drop_cols(Xf, names_f, NOENT)
            hgb_ne, _ = fit_hgb(Xf_ne, y, f_core, f_stop, fit)
            record("hgb_split_noent", hgb_ne.predict_proba(Xf_ne)[:, 1],
                   val, test, yt, taxes_val)

        if "hgb_split_nofieldtype" in scores_to_run:
            Xf_nf = drop_cols(Xf, names_f, fieldtype_cols(names_f))
            hgb_nf, _ = fit_hgb(Xf_nf, y, f_core, f_stop, fit)
            record("hgb_split_nofieldtype", hgb_nf.predict_proba(Xf_nf)[:, 1],
                   val, test, yt, taxes_val)

        if verbose and (sp + 1) % 10 == 0:
            print(f"    split {sp + 1}/{N_SPLITS} ({time.time() - t0:.0f}s)",
                  flush=True)

    out_scores: dict = {}
    for name, d in score_metrics.items():
        auroc = np.array(d["auroc"])
        entry = {
            "auroc_mean": round(float(auroc.mean()), 4),
            "auroc_std": round(float(auroc.std(ddof=1)), 4),
            "conformal": {},
        }
        for (a, tname), cell in d["cells"].items():
            cov = np.array(cell["cov"])
            risk = np.array(cell["risk"])
            entry["conformal"].setdefault(str(a), {})[tname] = {
                "coverage_mean": round(float(cov.mean()), 4),
                "coverage_std": round(float(cov.std(ddof=1)), 4),
                "risk_mean": round(float(risk.mean()), 4),
                "violation_frac": round(float((risk > a).mean()), 4),
                "held": bool(risk.mean() <= a),
                "cov_per_split": [round(c, 5) for c in cell["cov"]],
                "risk_per_split": [round(r, 5) for r in cell["risk"]],
            }
        out_scores[name] = entry

    return {
        "n_fields": len(recs),
        "n_docs": len(uniq),
        "base_rate_correct": round(float(y.mean()), 4),
        "hgb_best_iter_mean": (round(float(np.mean(hgb_iters)), 1)
                               if hgb_iters else None),
        "scores": out_scores,
    }


# ------------------------------------------------------------------- stats
def signflip_p(diff: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided sign-flip permutation p-value for mean(diff) = 0."""
    obs = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
    null = np.abs((flips * diff).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + N_PERM))


def cell_cov(ds_out, score, alpha, tax):
    return np.array(
        ds_out["scores"][score]["conformal"][str(alpha)][tax]["cov_per_split"]
    )


def paired_tests(ds_out: dict) -> dict:
    """(a) every grid cell vs baseline_lr pooled at the same alpha;
    (b) the subsumption/ablation contrasts. All paired over the 40 splits."""
    rng = np.random.default_rng(PERM_SEED)
    out: dict = {"vs_baseline_lr_pooled": {}, "contrasts": {}}
    for a in ALPHAS:
        ref = cell_cov(ds_out, "baseline_lr", a, "pooled")
        for name, e in ds_out["scores"].items():
            for tax in TAXONOMIES:
                if name == "baseline_lr" and tax == "pooled":
                    continue
                diff = cell_cov(ds_out, name, a, tax) - ref
                out["vs_baseline_lr_pooled"].setdefault(str(a), {})[
                    f"{name}|{tax}"
                ] = {
                    "mean_cov_diff": round(float(diff.mean()), 4),
                    "signflip_p": round(signflip_p(diff, rng), 5),
                }
    contrasts = [
        # does fieldtype-Mondrian add anything over hgb_split (score has FT)?
        ("subsume_freq", "hgb_split", "fieldtype-freq", "hgb_split", "pooled"),
        ("subsume_rule", "hgb_split", "fieldtype-rule", "hgb_split", "pooled"),
        # covariate in taxonomy instead of score: does it recover hgb_split?
        ("recover_freq", "hgb_split_nofieldtype", "fieldtype-freq",
         "hgb_split", "pooled"),
        ("recover_rule", "hgb_split_nofieldtype", "fieldtype-rule",
         "hgb_split", "pooled"),
        # within the ablated score, does fieldtype-Mondrian help at all?
        ("nft_freq_vs_nft_pooled", "hgb_split_nofieldtype", "fieldtype-freq",
         "hgb_split_nofieldtype", "pooled"),
        # cost of dropping fieldtype from the score entirely
        ("nofieldtype_cost", "hgb_split_nofieldtype", "pooled",
         "hgb_split", "pooled"),
        # cost of dropping entailment
        ("noent_cost_pooled", "hgb_split_noent", "pooled",
         "hgb_split", "pooled"),
        ("noent_cost_supbin", "hgb_split_noent", "support-bin",
         "hgb_split", "support-bin"),
    ]
    for a in ALPHAS:
        for label, s1, t1, s2, t2 in contrasts:
            diff = cell_cov(ds_out, s1, a, t1) - cell_cov(ds_out, s2, a, t2)
            out["contrasts"].setdefault(str(a), {})[label] = {
                "cell": f"{s1}|{t1} minus {s2}|{t2}",
                "mean_cov_diff": round(float(diff.mean()), 4),
                "signflip_p": round(signflip_p(diff, rng), 5),
            }
    return out


# -------------------------------------------------------------- sanity gate
def sanity_gate() -> dict:
    """Reproduce EXP-C hgb_split + baseline_lr on the OLD 6.9k CORD dump.
    Same split stream; only pooled/support-bin needed. Abort caller on fail."""
    recs = json.loads((OLD_DATA / "apivlm_perfield_rich_cord.json").read_text())
    print(f"  OLD cord: {len(recs)} fields, "
          f"{len(set(r['doc_id'] for r in recs))} docs", flush=True)
    out = run_dataset(recs, taxonomies=["pooled", "support-bin"],
                      scores_to_run=("baseline_lr", "hgb_split"), verbose=True)
    checks, ok = {}, True
    for (score, tax), (tc, tr) in SANITY_TARGETS.items():
        cell = out["scores"][score]["conformal"]["0.1"][tax]
        gc, gr = cell["coverage_mean"], cell["risk_mean"]
        passed = abs(gc - tc) <= SANITY_TOL and abs(gr - tr) <= SANITY_TOL
        ok &= passed
        checks[f"{score}|{tax}@0.10"] = {
            "expected_cov": tc, "got_cov": gc,
            "expected_risk": tr, "got_risk": gr, "pass": passed,
        }
    return {"pass": ok, "tolerance": SANITY_TOL, "checks": checks}


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    print("== SANITY GATE: reproduce EXP-C hgb_split on OLD cord (+-0.01) ==",
          flush=True)
    gate = sanity_gate()
    for k, c in gate["checks"].items():
        print(f"  {k}: cov {c['got_cov']:.4f} (exp {c['expected_cov']}) "
              f"risk {c['got_risk']:.4f} (exp {c['expected_risk']}) "
              f"-> {'PASS' if c['pass'] else 'FAIL'}", flush=True)
    if not gate["pass"]:
        Path(args.out).write_text(json.dumps(
            {"sanity_gate": gate, "aborted": True}, indent=1))
        print("SANITY GATE FAILED -- STOPPING (no scale runs).")
        sys.exit(1)
    print("  GATE PASS\n", flush=True)

    results = {
        "experiment": "EXP-F final score-side table on NEW 2x dumps",
        "seeds": {"split_seed": SPLIT_SEED, "sub_seed_base": SUB_SEED,
                  "inner_seed_base": INNER_SEED, "perm_seed": PERM_SEED,
                  "n_splits": N_SPLITS, "n_perm": N_PERM},
        "protocol": "40 doc-level 50/50 splits (seed 7); add-one selective-"
                    "risk threshold; split protocol for all learned scores "
                    "(model+transforms on cal fit-half, threshold+taxonomy "
                    "data on cal val-half); baseline_lr control additionally "
                    "under cal-protocol; nominal alpha, no band; risk 0-fill "
                    "when nothing accepted; taxonomies pooled/support-bin/"
                    "fieldtype-freq/fieldtype-rule with all data-dependent "
                    "parts from the threshold-fitting rows only.",
        "sanity_gate": gate,
        "datasets": {},
        "paired_tests": {},
    }
    for ds in DATASETS:
        recs = json.loads((NEW_DATA / f"apivlm_perfield_rich_{ds}.json").read_text())
        print(f"== NEW {ds}: n={len(recs)} fields, "
              f"{len(set(r['doc_id'] for r in recs))} docs ==", flush=True)
        ds_out = run_dataset(recs)
        results["datasets"][ds] = ds_out
        results["paired_tests"][ds] = paired_tests(ds_out)

        for name, e in sorted(ds_out["scores"].items(),
                              key=lambda kv: -kv[1]["auroc_mean"]):
            line = f"  {name:22} AUROC {e['auroc_mean']:.4f}"
            for tax in TAXONOMIES:
                c = e["conformal"]["0.1"][tax]
                line += (f" | {tax} {c['coverage_mean']:.3f}"
                         f"({c['risk_mean']:.3f}{'H' if c['held'] else 'X'})")
            print(line, flush=True)
        print(flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
