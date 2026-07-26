#!/usr/bin/env python
"""EXP-E: fixed + extended threshold harness -> FINAL money table.

Built from the validated EXP-B harness (exp_b/run_thresholds.py). Two changes,
each pinned by the root-cause analysis in exp_e/REPORT.md:

FIX 1 (candidate grid, fixes BOTH scale anomalies downstream):
  EXP-B built candidate thresholds as ts = sorted_scores[round(frac*n)-1].
  With '>= t' acceptance semantics this implicitly snaps every target to the
  smallest DISTINCT-VALUE boundary >= target ("snap up").  On a tie-heavy
  score (the scale CORD dump ships an all-zero `entailment` column, so the
  5-signal fusion collapses to ~257 distinct values over ~6.9k cal fields,
  tie masses of 200+), snap-up (a) collapses the 15 targets onto 6-8 distinct
  candidates via np.unique and (b) jumps the smallest candidate far past the
  intended acceptance (target 69 fields -> 245 accepted), skipping the entire
  certifiable low-risk head.  Downstream, LTT loses its only viable
  candidates and addone.doc's acceptance set goes empty -> exactly 0.0.
  # DECISION: snap each target count round(frac*n) to the NEAREST achievable
  distinct-value boundary count (ties -> the smaller, more conservative one).
  Label-independent (cal scores only, never labels) -> no validity cost, and
  on a tie-free score it is bit-identical to the EXP-B grid (regression mode
  verifies this empirically; a synthetic self-test below pins the semantics).

FIX 2 (addone.doc): no rule change -- its exact-0.0 collapse was the empty
  acceptance set produced by FIX 1's pathology (min add-one doc bound 0.24 at
  alpha=0.10 because the coarsest reachable candidate already accepted 245
  fields / 71 docs).  On the fixed grid the method is exercised again.

EXTENSION: schema-derived field-type Mondrian taxonomies from EXP-A
  (cal-only vocabularies), the scale winners:
    ftf = fieldtype-freq : leaf key of the field path (list indices stripped,
          last dotted component, lowercased); keys with >= 25 CAL fields get
          their own group, everything else -> group 0.  Vocabulary rebuilt on
          the calibration half of every split.
    ftr = fieldtype-rule : fixed keyword rule (date>contact>amount>count>name
          >other), no data dependence at all.
  Final taxonomy set: pooled | sb (support-bin, cal-quantile .34/.67) | ftf |
  ftr.  Unseen/empty groups on test reject (EXP-A semantics).

Modes (one invocation runs both):
  regression : OLD dumps (repo data/), EXP-B taxonomy+procedure set, EXP-B
               sanity gate (addone pooled/gxsfull vs phase2_method.py), then a
               cell-by-cell diff against exp_b/results.json.  Scale runs are
               gated on this passing.
  scale      : NEW 2x dumps (exp_d/data), final taxonomy set, full procedure
               grid, money table + paired sign-flip tests vs pooled.

Honest-evaluation contract (unchanged): 40 doc-level 50/50 splits, seed 7,
NOMINAL alpha only, held iff mean achieved test risk <= alpha (no tolerance),
violation fraction over the 40 splits, no test tuning, cal-only vocabularies
and bin edges.  Guarantee forms as in EXP-B (see GUARANTEES below).

KNOWN DATA DEFECT (root cause of anomaly 1, documented, NOT silently fixed):
the scale CORD dump's `entailment` column is identically 0.0 (OLD: 2,858
distinct values; scale FUNSD/XFUND are intact).  Scale-CORD results therefore
measure a 4-signal fusion and are not comparable to OLD-CORD absolute levels.

Usage: python run_final.py [--splits 40] [--out results.json] [--skip-regression]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy import special, stats
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
OLD_DATA_DIR = Path("archive/pre-campaign-2026-07-26/data")
NEW_DATA_DIR = Path("data")
EXPB_RESULTS = HERE.parent / "exp_b" / "results.json"

DATASETS = ["cord", "funsd", "xfund"]
SIGNAL_KEYS = ["verbalized", "consistency", "support", "grounded", "entailment"]
N_SPLITS = 40
SEED = 7            # must match scripts/phase2_method.py
AUX_SEED = 12345    # betting doc order
PERM_SEED = 123     # sign-flip permutation tests
N_PERM = 20000
DELTA = 0.10        # FWER / confidence level for all LTT variants
ALPHAS = [0.05, 0.10, 0.15, 0.20]
HEADLINE_ALPHAS = [0.05, 0.10]
TAXES_EXPB = ["pooled", "sb", "sxv", "gxs", "cb"]
TAXES_FINAL = ["pooled", "sb", "ftf", "ftr"]
FT_MIN_CAL = 25     # EXP-A fieldtype-freq vocabulary threshold

Q_FRAC = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075,
                   0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00])

SANITY = {"pooled": (0.056, 0.105), "gxsfull": (0.114, 0.122)}
SANITY_TOL = 0.005
# Regression gate for the grid fix.  The fix is bit-identical on tie-free
# scores but by DESIGN moves LTT candidates on tie-affected subsets (that is
# the fix), so cell means may drift.  Gate: (a) headline alphas 0.05/0.10 may
# drift at most 0.01 absolute coverage, (b) at any alpha the drift must stay
# well inside split-to-split noise (<= 0.5 of the EXP-B cell's cov_std).
REGRESSION_TOL_HEADLINE = 0.01
REGRESSION_TOL_NOISE = 0.5

GUARANTEES = {
    "addone": "E[selective risk] <= alpha, marginal, IF fields exchangeable (known broken here: doc clustering + threshold optimization + score refit)",
    "addone.doc": "heuristic CRC add-one on per-doc loss over accepting docs (loss non-monotone in t); macro target",
    "ltt.binom/hoeffKL": f"P(field selective risk > alpha) <= {DELTA} IF accepted-field errors iid (doc clustering ignored)",
    "ltt.neff": f"approx P(field selective risk > alpha) <= {DELTA}, plug-in cluster design effect",
    "ltt.dochb/docbet": f"P(mean per-doc selective error rate among accepting docs > alpha) <= {DELTA}, finite-sample, docs iid",
}


# --------------------------------------------------------------------------- #
# Reference machinery (verbatim from EXP-B / phase2_method.py)
# --------------------------------------------------------------------------- #
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


def cal_bins(x: np.ndarray, cal: np.ndarray, q) -> np.ndarray:
    edges = np.unique(np.quantile(x[cal], q))
    if len(edges) == 0:
        return np.zeros(len(x), int)
    return np.digitize(x, edges)


# ---- EXP-A fieldtype taxonomies (verbatim semantics) ---------------------- #
def leaf_key(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path).split(".")[-1].lower()


def field_type_rule(path: str) -> int:
    """Fixed keyword rule (no data dependence -> trivially leakage-free).
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


def make_taxonomies(sig: dict, paths: list[str], cal: np.ndarray) -> dict[str, np.ndarray]:
    g = sig["grounded"].astype(int)
    sb = cal_bins(sig["support"], cal, (0.34, 0.67))
    vb = cal_bins(sig["verbalized"], cal, (0.5,))
    cb = cal_bins(sig["consistency"], cal, (0.5,))
    sb_full = np.digitize(sig["support"], np.quantile(sig["support"], (0.34, 0.67)))
    lks = np.array([leaf_key(p) for p in paths])
    cal_keys, cal_counts = np.unique(lks[cal], return_counts=True)
    vocab = {k: i + 1 for i, k in enumerate(sorted(cal_keys[cal_counts >= FT_MIN_CAL]))}
    return {
        "pooled": np.zeros(len(g), int),
        "sb": sb,
        "sxv": sb * 2 + vb,
        "gxs": g * 3 + sb,
        "cb": cb,
        "gxsfull": g * 3 + sb_full,                       # sanity gate only
        "ftf": np.array([vocab.get(k, 0) for k in lks], int),
        "ftr": np.array([field_type_rule(p) for p in paths], int),
    }


# --------------------------------------------------------------------------- #
# p-values for H0: selective risk R(t) > alpha (verbatim from EXP-B)
# --------------------------------------------------------------------------- #
def p_binom_exact(k, n, alpha):
    p = np.ones_like(k, dtype=float)
    pos = n > 0
    p[pos] = stats.binom.cdf(k[pos], n[pos], alpha)
    return np.clip(p, 0.0, 1.0)


def _kl_bern(a, b):
    a = np.clip(a, 1e-12, 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def p_hoeffding_kl(lsum, n, alpha):
    p = np.ones_like(lsum, dtype=float)
    pos = n > 0
    rhat = np.zeros_like(lsum, dtype=float)
    rhat[pos] = lsum[pos] / n[pos]
    lo = pos & (rhat < alpha)
    p[lo] = np.exp(-n[lo] * _kl_bern(rhat[lo], alpha))
    return np.clip(p, 0.0, 1.0)


def p_bentkus(lsum, n, alpha):
    p = np.ones_like(lsum, dtype=float)
    pos = n > 0
    p[pos] = math.e * stats.binom.cdf(np.ceil(lsum[pos]), n[pos], alpha)
    return np.clip(p, 0.0, 1.0)


def p_hb(lsum, n, alpha):
    return np.minimum(p_hoeffding_kl(lsum, n, alpha), p_bentkus(lsum, n, alpha))


def p_binom_neff(k, n, kd_sq, ndocs, alpha):
    p = np.ones_like(k, dtype=float)
    pos = n > 0
    rhat = np.zeros_like(k, dtype=float)
    rhat[pos] = k[pos] / n[pos]
    v = np.zeros_like(k, dtype=float)
    v[pos] = kd_sq[pos] / np.maximum(n[pos], 1) ** 2
    n_eff = np.where(v > 0, rhat * (1 - rhat) / np.maximum(v, 1e-300), 0.0)
    n_eff = np.minimum(np.where((rhat <= 0) | (rhat >= 1) | (v <= 0), ndocs, n_eff), n)
    n_eff = np.maximum(n_eff, 1e-9)
    k_eff = rhat * n_eff
    lo = pos & (rhat < alpha)
    p[lo] = special.betainc(np.maximum(n_eff[lo] - k_eff[lo], 1e-9), k_eff[lo] + 1.0, 1 - alpha)
    return np.clip(p, 0.0, 1.0)


def betting_pvals(loss_mat, alpha, delta):
    D, m = loss_mat.shape
    logw = np.zeros(m)
    maxlogw = np.zeros(m)
    cnt = np.zeros(m)
    s_sum = np.zeros(m)
    q_sum = np.zeros(m)
    lam_cap = 0.5 / (1.0 - alpha)
    log_inv_delta = math.log(1.0 / delta)
    for i in range(D):
        row = loss_mat[i]
        msk = ~np.isnan(row)
        if not msk.any():
            continue
        mu_hat = (0.5 + s_sum[msk]) / (cnt[msk] + 1.0)
        sig2 = (0.25 + q_sum[msk]) / (cnt[msk] + 1.0)
        lam = np.minimum(np.sqrt(2.0 * log_inv_delta / (D * sig2)), lam_cap)
        x = row[msk]
        logw[msk] += np.log1p(lam * (alpha - x))
        maxlogw[msk] = np.maximum(maxlogw[msk], logw[msk])
        q_sum[msk] += (x - mu_hat) ** 2
        s_sum[msk] += x
        cnt[msk] += 1.0
    p = np.exp(-maxlogw)
    p[cnt == 0] = 1.0
    return np.clip(p, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Cal-side sufficient statistics -- with the FIXED candidate grid
# --------------------------------------------------------------------------- #
def snap_targets_to_boundaries(ss_desc: np.ndarray, fracs: np.ndarray) -> np.ndarray:
    """FIX 1. Achievable acceptance counts are the cumulative counts at
    DISTINCT-value boundaries of the descending-sorted cal scores.  Snap each
    target count round(frac*n) to the NEAREST achievable count (ties -> the
    smaller / more conservative).  Tie-free scores reproduce the EXP-B grid
    exactly; label-independent by construction."""
    n = len(ss_desc)
    bnd = np.flatnonzero(np.append(ss_desc[:-1] > ss_desc[1:], True)) + 1.0
    targets = np.unique(np.clip(np.round(fracs * n).astype(int), 1, n)).astype(float)
    pos = np.searchsorted(bnd, targets)          # first boundary >= target
    hi = np.minimum(pos, len(bnd) - 1)
    lo = np.maximum(pos - 1, 0)
    up_closer = (bnd[hi] - targets) < (targets - bnd[lo])
    take = np.where((pos == 0) | up_closer, hi, lo)
    return np.unique(bnd[take]).astype(int)      # ascending acceptance counts


class CalStats:
    """Cal-side statistics for one field subset at the fixed label-independent
    candidate grid (boundary-snapped acceptance-fraction quantiles)."""

    def __init__(self, s, y, dloc, ndoc, doc_order, fracs=Q_FRAC):
        self.n_cal = len(s)
        order = np.argsort(-s, kind="stable")
        ss, yy, dd = s[order], y[order], dloc[order]
        if self.n_cal == 0:
            self.ts = np.array([np.inf])
        else:
            idx = snap_targets_to_boundaries(ss, fracs)
            self.ts = ss[idx - 1]                # distinct, descending
        self.m = len(self.ts)
        cum_err = np.cumsum(1.0 - yy) if self.n_cal else np.zeros(0)
        self.n_t = np.searchsorted(-ss, -self.ts, side="right")
        self.k_t = np.where(self.n_t > 0, cum_err[np.maximum(self.n_t - 1, 0)], 0.0) \
            if self.n_cal else np.zeros(self.m)
        self.acc = np.zeros((self.m, ndoc))
        self.err = np.zeros((self.m, ndoc))
        e = 1.0 - yy
        for j in range(self.m):
            c = self.n_t[j]
            if c > 0:
                self.acc[j] = np.bincount(dd[:c], minlength=ndoc)
                self.err[j] = np.bincount(dd[:c], weights=e[:c], minlength=ndoc)
        with np.errstate(invalid="ignore", divide="ignore"):
            loss = np.where(self.acc > 0, self.err / np.maximum(self.acc, 1), np.nan)
        self.loss_mat = loss[:, doc_order].T
        self.ndocs_t = (self.acc > 0).sum(axis=1).astype(float)
        self.docloss_sum = np.nansum(loss, axis=1)
        rhat = np.where(self.n_t > 0, self.k_t / np.maximum(self.n_t, 1), 0.0)
        self.kd_sq = ((self.err - self.acc * rhat[:, None]) ** 2).sum(axis=1)
        self._pcache: dict = {}

    def pvec(self, pfun: str, alpha: float) -> np.ndarray:
        key = (pfun, alpha)
        if key not in self._pcache:
            if pfun == "binom":
                p = p_binom_exact(self.k_t, self.n_t, alpha)
            elif pfun == "hoeffKL":
                p = p_hoeffding_kl(self.k_t, self.n_t.astype(float), alpha)
            elif pfun == "neff":
                p = p_binom_neff(self.k_t, self.n_t, self.kd_sq, self.ndocs_t, alpha)
            elif pfun == "dochb":
                p = p_hb(self.docloss_sum, self.ndocs_t, alpha)
            elif pfun == "docbet":
                p = betting_pvals(self.loss_mat, alpha, DELTA)
            else:
                raise ValueError(pfun)
            self._pcache[key] = p
        return self._pcache[key]

    def nvec(self, pfun: str) -> np.ndarray:
        return self.n_t.astype(float) if pfun in ("binom", "hoeffKL") else self.ndocs_t


def _selftest_grid():
    """Pin FIX 1 semantics: tie-free == EXP-B grid; ties snap to nearest
    boundary, ties resolved to the smaller count."""
    n = 500
    rng = np.random.default_rng(0)
    ss = np.sort(rng.random(n))[::-1]            # tie-free
    idx = snap_targets_to_boundaries(ss, Q_FRAC)
    want = np.unique(np.clip(np.round(Q_FRAC * n).astype(int), 1, n))
    assert np.array_equal(idx, want), "tie-free grid must equal EXP-B grid"
    ss2 = np.array([.9] * 3 + [.8] * 2 + [.7] * 5)   # boundaries at 3,5,10
    idx2 = snap_targets_to_boundaries(ss2, np.array([0.2, 0.4, 0.65, 1.0]))
    # targets 2,4,6,10 -> nearest boundaries 3, 3 (tie 3/5 -> smaller), 5, 10
    assert idx2.tolist() == [3, 5, 10], idx2
    # EXP-B snap-up would have produced [3, 5, 10] here too, but for target 4
    # it takes 5 while we take 3 (tie -> conservative); check directly:
    one = snap_targets_to_boundaries(ss2, np.array([0.4]))
    assert one.tolist() == [3]


_selftest_grid()


# --------------------------------------------------------------------------- #
# Selection rules (verbatim from EXP-B)
# --------------------------------------------------------------------------- #
def holm_select(cs, pvals, budget):
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    levels = budget / (m - np.arange(m))
    ok = pvals[order] <= levels
    first_fail = int(np.argmax(~ok)) if (~ok).any() else m
    if first_fail == 0:
        return np.inf
    rej = order[:first_fail]
    return float(cs.ts[rej[np.argmax(cs.n_t[rej])]])


def fs_select(cs, pvals, nvec, budget, alpha):
    pmin = (1.0 - alpha) ** np.maximum(nvec, 0.0)
    best = None
    started = False
    for j in range(cs.m):
        if not started:
            if pmin[j] > budget:
                continue
            started = True
        if pvals[j] <= budget:
            best = j
        else:
            break
    return float(cs.ts[best]) if best is not None else np.inf


def ltt_select(cs, alpha, budget, pfun, rule):
    p = cs.pvec(pfun, alpha)
    if rule == "holm":
        return holm_select(cs, p, budget)
    if rule == "fs":
        return fs_select(cs, p, cs.nvec(pfun), budget, alpha)
    if rule == "mix":
        t1 = holm_select(cs, p, budget / 2)
        t2 = fs_select(cs, p, cs.nvec(pfun), budget / 2, alpha)
        cands = [t for t in (t1, t2) if np.isfinite(t)]
        if not cands:
            return np.inf
        return float(min(cands, key=lambda t: -cs.n_t[int(np.argmin(np.abs(cs.ts - t)))]))
    raise ValueError(rule)


def doc_addone_select(cs, alpha):
    with np.errstate(invalid="ignore", divide="ignore"):
        bound = (cs.docloss_sum + 1.0) / (cs.ndocs_t + 1.0)
    ok = np.where((cs.ndocs_t > 0) & (bound <= alpha))[0]
    if len(ok) == 0:
        return np.inf
    return float(cs.ts[ok[np.argmax(cs.n_t[ok])]])


def grouped(select_fn, cs_by_group, budget_total, per_group):
    G = len(cs_by_group)
    if G == 0:
        return {}
    b = budget_total if per_group else budget_total / G
    return {g: select_fn(cs, b) for g, cs in cs_by_group.items()}


def evaluate(score_t, y_t, groups_t, tdict, dloc_t, ndoc):
    acc = np.zeros(len(score_t), bool)
    for g, t in tdict.items():
        if np.isfinite(t):
            acc |= (groups_t == g) & (score_t >= t)
    cov = float(acc.mean()) if len(acc) else 0.0
    risk = float(1.0 - y_t[acc].mean()) if acc.any() else 0.0
    if acc.any():
        a = np.bincount(dloc_t[acc], minlength=ndoc)
        e = np.bincount(dloc_t[acc], weights=1.0 - y_t[acc], minlength=ndoc)
        nz = a > 0
        docrisk = float(np.mean(e[nz] / a[nz]))
    else:
        docrisk = 0.0
    return cov, risk, docrisk, int(acc.sum())


def signflip_p(diff, rng):
    obs = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
    null = np.abs((flips * diff).mean(axis=1))
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + N_PERM))


# --------------------------------------------------------------------------- #
# Procedure grids
# --------------------------------------------------------------------------- #
LTT_RUNS_EXPB = [
    ("ltt.binom.holm", "binom", "holm", TAXES_EXPB, False),
    ("ltt.binom.holm.pg", "binom", "holm", ["sb", "sxv", "gxs", "cb"], True),
    ("ltt.binom.fs", "binom", "fs", TAXES_EXPB, False),
    ("ltt.binom.fs.pg", "binom", "fs", ["sb", "sxv", "gxs", "cb"], True),
    ("ltt.binom.mix", "binom", "mix", ["pooled", "sb"], False),
    ("ltt.binom.mix.pg", "binom", "mix", ["sb", "sxv", "cb"], True),
    ("ltt.hoeffKL.holm", "hoeffKL", "holm", ["pooled", "sb"], False),
    ("ltt.neff.holm", "neff", "holm", TAXES_EXPB, False),
    ("ltt.neff.holm.pg", "neff", "holm", ["sb"], True),
    ("ltt.dochb.holm", "dochb", "holm", TAXES_EXPB, False),
    ("ltt.dochb.holm.pg", "dochb", "holm", ["sb", "sxv", "gxs", "cb"], True),
    ("ltt.dochb.fs.pg", "dochb", "fs", ["sb"], True),
    ("ltt.docbet.holm", "docbet", "holm", ["pooled", "sb"], False),
    ("ltt.docbet.holm.pg", "docbet", "holm", ["sb"], True),
]
NONPOOL_FINAL = ["sb", "ftf", "ftr"]
LTT_RUNS_FINAL = [
    ("ltt.binom.holm", "binom", "holm", TAXES_FINAL, False),
    ("ltt.binom.holm.pg", "binom", "holm", NONPOOL_FINAL, True),
    ("ltt.binom.fs", "binom", "fs", TAXES_FINAL, False),
    ("ltt.binom.fs.pg", "binom", "fs", NONPOOL_FINAL, True),
    ("ltt.binom.mix", "binom", "mix", TAXES_FINAL, False),
    ("ltt.binom.mix.pg", "binom", "mix", NONPOOL_FINAL, True),
    ("ltt.hoeffKL.holm", "hoeffKL", "holm", ["pooled", "sb"], False),
    ("ltt.neff.holm.pg", "neff", "holm", ["sb", "ftf"], True),
    ("ltt.dochb.holm", "dochb", "holm", TAXES_FINAL, False),
    ("ltt.dochb.holm.pg", "dochb", "holm", NONPOOL_FINAL, True),
    ("ltt.docbet.holm", "docbet", "holm", ["pooled", "sb"], False),
    ("ltt.docbet.holm.pg", "docbet", "holm", ["sb", "ftf"], True),
]


def load(data_dir: Path, ds: str):
    recs = json.loads((data_dir / f"apivlm_perfield_rich_{ds}.json").read_text())
    docs = np.array([r["doc_id"] for r in recs])
    y = np.array([r["correct"] for r in recs], float)
    X = np.column_stack([np.array([float(r.get(k, 0.0)) for r in recs]) for k in SIGNAL_KEYS])
    sig = {k: X[:, i] for i, k in enumerate(SIGNAL_KEYS)}
    paths = [r["path"] for r in recs]
    return docs, y, X, sig, paths


def run_dataset(data_dir: Path, ds: str, n_splits: int, mode: str):
    docs, y, X, sig, paths = load(data_dir, ds)
    uniq = list(dict.fromkeys(docs.tolist()))
    doc_code = {d: i for i, d in enumerate(uniq)}
    dloc_all = np.array([doc_code[d] for d in docs])
    ndoc = len(uniq)
    rng = np.random.default_rng(SEED)              # identical stream to EXP-B
    rng_aux = np.random.default_rng(SEED + AUX_SEED)

    taxes_addone = (TAXES_EXPB + ["gxsfull"]) if mode == "regression" else TAXES_FINAL
    taxes_cs = TAXES_EXPB if mode == "regression" else TAXES_FINAL
    taxes_docaddone = ["pooled", "sb"] if mode == "regression" else TAXES_FINAL
    ltt_runs = LTT_RUNS_EXPB if mode == "regression" else LTT_RUNS_FINAL

    out: dict = {}

    def rec(proc, tax, alpha, cov, risk, docrisk, nacc):
        d = out.setdefault((proc, tax, alpha),
                           {"cov": [], "risk": [], "docrisk": [], "nacc": []})
        d["cov"].append(cov)
        d["risk"].append(risk)
        d["docrisk"].append(docrisk)
        d["nacc"].append(nacc)

    for _sp in range(n_splits):
        perm = rng.permutation(uniq)
        half = len(uniq) // 2
        cal_docs = set(perm[:half].tolist())
        cal = np.array([d in cal_docs for d in docs])
        test = ~cal

        lr = LogisticRegression(max_iter=1000).fit(X[cal], y[cal])
        fusion = lr.predict_proba(X)[:, 1]

        taxes = make_taxonomies(sig, paths, cal)
        s_cal, y_cal, dloc_cal = fusion[cal], y[cal], dloc_all[cal]
        s_tst, y_tst, dloc_tst = fusion[test], y[test], dloc_all[test]
        doc_order = rng_aux.permutation(ndoc)

        cs_tax: dict[str, dict[int, CalStats]] = {}
        for tname in taxes_cs:
            gr_cal = taxes[tname][cal]
            cs_tax[tname] = {
                g: CalStats(s_cal[gr_cal == g], y_cal[gr_cal == g],
                            dloc_cal[gr_cal == g], ndoc, doc_order)
                for g in np.unique(gr_cal)
            }

        for alpha in ALPHAS:
            for tname in taxes_addone:
                gr = taxes[tname]
                gr_cal2 = gr[cal]
                tdict = {
                    g: conformal_threshold(s_cal[gr_cal2 == g], y_cal[gr_cal2 == g], alpha)
                    for g in np.unique(gr_cal2)
                }
                rec("addone", tname, alpha,
                    *evaluate(s_tst, y_tst, gr[test], tdict, dloc_tst, ndoc))

            for tname in taxes_docaddone:
                tdict = {g: doc_addone_select(cs, alpha) for g, cs in cs_tax[tname].items()}
                rec("addone.doc", tname, alpha,
                    *evaluate(s_tst, y_tst, taxes[tname][test], tdict, dloc_tst, ndoc))

            for proc, pfun, rule, tax_list, per_group in ltt_runs:
                for tname in tax_list:
                    tdict = grouped(
                        lambda cs, b: ltt_select(cs, alpha, b, pfun, rule),
                        cs_tax[tname], DELTA, per_group)
                    rec(proc, tname, alpha,
                        *evaluate(s_tst, y_tst, taxes[tname][test], tdict, dloc_tst, ndoc))

    agg: dict = {}
    for (proc, tax, alpha), d in out.items():
        cov = np.array(d["cov"])
        risk = np.array(d["risk"])
        docr = np.array(d["docrisk"])
        agg.setdefault(str(alpha), {}).setdefault(proc, {})[tax] = {
            "cov_mean": round(float(cov.mean()), 4),
            "cov_std": round(float(cov.std(ddof=1)), 4),
            "risk_mean": round(float(risk.mean()), 4),
            "viol_frac": round(float((risk > alpha).mean()), 4),
            "held": bool(risk.mean() <= alpha),
            "docrisk_mean": round(float(docr.mean()), 4),
            "viol_frac_doc": round(float((docr > alpha).mean()), 4),
            "held_doc": bool(docr.mean() <= alpha),
            "zero_cov_frac": round(float((cov == 0).mean()), 4),
            "nacc_mean": round(float(np.mean(d["nacc"])), 1),
            "cov_per_split": [round(c, 5) for c in d["cov"]],
            "risk_per_split": [round(r, 5) for r in d["risk"]],
        }
    return agg


# --------------------------------------------------------------------------- #
# Regression check vs EXP-B results.json
# --------------------------------------------------------------------------- #
def regression_check(results_old: dict) -> dict:
    expb = json.loads(EXPB_RESULTS.read_text())["results"]
    diffs = []
    for ds in DATASETS:
        for a, by_proc in results_old[ds].items():
            for proc, by_tax in by_proc.items():
                for tax, cell in by_tax.items():
                    ref = expb.get(ds, {}).get(a, {}).get(proc, {}).get(tax)
                    if ref is None:
                        continue
                    dcov = cell["cov_mean"] - ref["cov_mean"]
                    diffs.append({
                        "cell": f"{ds}/{a}/{proc}/{tax}", "alpha": a,
                        "d_cov": round(dcov, 4),
                        "d_risk": round(cell["risk_mean"] - ref["risk_mean"], 4),
                        "d_cov_over_std": round(abs(dcov) / max(ref["cov_std"], 1e-4), 3),
                        "cov_new": cell["cov_mean"], "cov_expb": ref["cov_mean"],
                    })
    max_dcov = max(abs(d["d_cov"]) for d in diffs)
    max_dcov_headline = max(abs(d["d_cov"]) for d in diffs
                            if d["alpha"] in ("0.05", "0.1"))
    max_norm = max(d["d_cov_over_std"] for d in diffs)
    exact = sum(1 for d in diffs if d["d_cov"] == 0 and d["d_risk"] == 0)
    moved = sorted((d for d in diffs if abs(d["d_cov"]) > 0.002),
                   key=lambda d: -abs(d["d_cov"]))
    return {
        "n_cells_compared": len(diffs),
        "n_cells_exactly_equal": exact,
        "max_abs_d_cov": max_dcov,
        "max_abs_d_cov_headline_alphas": max_dcov_headline,
        "max_abs_d_cov_over_expb_std": max_norm,
        "pass": (max_dcov_headline <= REGRESSION_TOL_HEADLINE
                 and max_norm <= REGRESSION_TOL_NOISE),
        "gate": {"headline_abs_tol": REGRESSION_TOL_HEADLINE,
                 "noise_normalized_tol": REGRESSION_TOL_NOISE},
        "cells_moved_gt_0.002": moved[:20],
    }


def sanity_gate(results_old: dict) -> bool:
    ok_all = True
    for tax, want in SANITY.items():
        got = results_old["cord"]["0.1"]["addone"][tax]
        ok = (abs(got["cov_mean"] - want[0]) <= SANITY_TOL
              and abs(got["risk_mean"] - want[1]) <= SANITY_TOL)
        ok_all &= ok
        print(f"SANITY addone.{tax}: cov {got['cov_mean']:.4f} (want {want[0]}), "
              f"risk {got['risk_mean']:.4f} (want {want[1]}) -> {'PASS' if ok else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------- #
# Money table
# --------------------------------------------------------------------------- #
ADDONE_FAMILY = ["addone", "addone.doc"]
LTT_FIELD = ["ltt.binom.holm", "ltt.binom.holm.pg", "ltt.binom.fs", "ltt.binom.fs.pg",
             "ltt.binom.mix", "ltt.binom.mix.pg", "ltt.hoeffKL.holm"]
LTT_DOC = ["ltt.dochb.holm", "ltt.dochb.holm.pg", "ltt.docbet.holm", "ltt.docbet.holm.pg"]
POOLED_BASE = {p: p for p in ADDONE_FAMILY}          # same-procedure pooled baseline
for p in LTT_FIELD:
    POOLED_BASE[p] = "ltt.binom.holm" if p.startswith("ltt.binom") else "ltt.hoeffKL.holm"
for p in LTT_DOC:
    POOLED_BASE[p] = "ltt.dochb.holm" if "dochb" in p else "ltt.docbet.holm"


def cell_summary(cell):
    return {"cov_mean": cell["cov_mean"], "cov_std": cell["cov_std"],
            "risk_mean": cell["risk_mean"], "viol_frac": cell["viol_frac"],
            "held": cell["held"], "nacc_mean": cell["nacc_mean"]}


def build_money_table(results_scale: dict) -> dict:
    rngp = np.random.default_rng(PERM_SEED)
    table: dict = {}
    for ds in DATASETS:
        for a in map(str, HEADLINE_ALPHAS):
            grid = results_scale[ds][a]
            alpha = float(a)

            def best(procs):
                cands = []
                for p in procs:
                    for tax, cell in grid.get(p, {}).items():
                        if cell["held"]:
                            cands.append((cell["cov_mean"], p, tax, cell))
                if not cands:
                    return None
                cands.sort(key=lambda c: -c[0])
                return cands[0]

            def attach_p(pick):
                if pick is None:
                    return None
                cov, proc, tax, cell = pick
                basep = POOLED_BASE.get(proc, "addone")
                base = grid.get(basep, {}).get("pooled")
                entry = {"procedure": proc, "taxonomy": tax, **cell_summary(cell)}
                if base is not None and (proc, tax) != (basep, "pooled"):
                    diff = (np.array(cell["cov_per_split"])
                            - np.array(base["cov_per_split"]))
                    p = signflip_p(diff, rngp)
                    entry["vs_pooled"] = {
                        "baseline": f"{basep}/pooled",
                        "cov_diff_mean": round(float(diff.mean()), 4),
                        "signflip_p": round(p, 5),
                        "significant_0.05": bool(p < 0.05 and diff.mean() > 0),
                    }
                return entry

            best_addone = attach_p(best(ADDONE_FAMILY))
            best_ltt = attach_p(best(LTT_FIELD + LTT_DOC))
            best_ltt_doc = attach_p(best(LTT_DOC))
            if best_ltt is not None:
                fam = ("ltt.dochb/docbet" if any(k in best_ltt["procedure"]
                       for k in ("dochb", "docbet"))
                       else "ltt.neff" if "neff" in best_ltt["procedure"]
                       else "ltt.binom/hoeffKL")
                best_ltt["guarantee"] = GUARANTEES[fam]
            if best_ltt_doc is not None:
                best_ltt_doc["guarantee"] = GUARANTEES["ltt.dochb/docbet"]
            pooled = {p: cell_summary(grid[p]["pooled"])
                      for p in ("addone", "addone.doc", "ltt.binom.holm", "ltt.dochb.holm")
                      if p in grid and "pooled" in grid[p]}
            table.setdefault(ds, {})[a] = {
                "best_addone_family_held": best_addone,
                "best_ltt_held": best_ltt,
                "best_ltt_doclevel_held": best_ltt_doc,
                "pooled_baselines": pooled,
            }
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--skip-regression", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    payload: dict = {"config": {
        "n_splits": args.splits, "seed": SEED, "delta_fwer": DELTA,
        "alphas": ALPHAS, "q_frac": Q_FRAC.tolist(), "signal_keys": SIGNAL_KEYS,
        "taxonomies_final": TAXES_FINAL, "ft_min_cal": FT_MIN_CAL,
        "grid_fix": "candidate thresholds snapped to distinct-value boundaries "
                    "nearest round(frac*n_cal), ties -> smaller (EXP-E FIX 1)",
        "budgets": {"default": "DELTA/G per group (simultaneous)",
                    ".pg": "DELTA per group (per-group statement)"},
        "guarantees": GUARANTEES,
        "known_data_defect": "scale CORD dump: `entailment` identically 0.0 "
                             "(OLD had 2,858 distinct values); scale CORD is a "
                             "4-signal fusion, not comparable to OLD absolute levels",
    }}

    # ---------------- regression on OLD data ------------------------------ #
    if not args.skip_regression:
        print("=== regression mode: OLD dumps, EXP-B procedure set, fixed grid ===")
        res_old = {}
        for ds in DATASETS:
            t = time.time()
            res_old[ds] = run_dataset(OLD_DATA_DIR, ds, args.splits, "regression")
            print(f"[regression {ds}] done in {time.time() - t:.1f}s")
        gate = sanity_gate(res_old) if args.splits == N_SPLITS else None
        reg = regression_check(res_old)
        print(f"regression: {reg['n_cells_compared']} cells, "
              f"{reg['n_cells_exactly_equal']} exactly equal, "
              f"max|dcov|={reg['max_abs_d_cov']} "
              f"(headline {reg['max_abs_d_cov_headline_alphas']}, "
              f"max/std {reg['max_abs_d_cov_over_expb_std']}) "
              f"-> {'PASS' if reg['pass'] else 'FAIL'}")
        payload["regression_check"] = {"sanity_gate_pass": gate, **reg}
        # keep the anomalous OLD cells for the report
        payload["regression_anomalous_cells_old"] = {
            "ltt.binom.mix.pg/sb@0.1": res_old["cord"]["0.1"]["ltt.binom.mix.pg"]["sb"]["cov_mean"],
            "addone.doc/sb@0.1": res_old["cord"]["0.1"]["addone.doc"]["sb"]["cov_mean"],
            "addone.doc/pooled@0.1": res_old["cord"]["0.1"]["addone.doc"]["pooled"]["cov_mean"],
        }
        if gate is False or not reg["pass"]:
            print("REGRESSION FAILED -- not running scale.")
            Path(args.out).write_text(json.dumps(payload, indent=1))
            raise SystemExit(1)

    # ---------------- scale run on NEW data ------------------------------- #
    print("\n=== scale mode: NEW 2x dumps, final taxonomy set ===")
    res_scale = {}
    for ds in DATASETS:
        t = time.time()
        res_scale[ds] = run_dataset(NEW_DATA_DIR, ds, args.splits, "scale")
        print(f"[scale {ds}] done in {time.time() - t:.1f}s")
    payload["results_scale"] = res_scale
    payload["money_table"] = build_money_table(res_scale)

    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}  ({time.time() - t0:.1f}s total)")

    # console money table
    for ds in DATASETS:
        for a in map(str, HEADLINE_ALPHAS):
            m = payload["money_table"][ds][a]
            print(f"\n=== MONEY {ds} alpha={a} ===")
            for key in ("best_addone_family_held", "best_ltt_held",
                        "best_ltt_doclevel_held"):
                e = m[key]
                if e is None:
                    print(f"  {key:28} -- none held")
                    continue
                vp = e.get("vs_pooled", {})
                print(f"  {key:28} {e['procedure']}/{e['taxonomy']}: "
                      f"cov {e['cov_mean']:.4f}±{e['cov_std']:.4f} "
                      f"risk {e['risk_mean']:.4f} viol {e['viol_frac']:.2f} "
                      f"p_vs_pooled={vp.get('signflip_p', '--')}")
            for p, c in m["pooled_baselines"].items():
                print(f"  pooled[{p:14}] cov {c['cov_mean']:.4f}±{c['cov_std']:.4f} "
                      f"risk {c['risk_mean']:.4f} viol {c['viol_frac']:.2f} held={c['held']}")


if __name__ == "__main__":
    main()
