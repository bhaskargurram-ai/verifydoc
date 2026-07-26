#!/usr/bin/env python3
"""Build the human-annotation package (issue #68): 600 triple-annotated items.

Design (one study, two upgrades):
  (a) label reliability: Fleiss/Cohen kappa on human labels + kappa(auto, human)
      -> Paper B's #1 weakness;
  (b) human-verified selective risk: 150 items sampled from the ACCEPTED set of
      the production trust config (hgb_split_noent recipe, split protocol,
      add-one @ alpha=0.10, seed-7 doc split) -> validates Paper A's guarantee
      against label noise.

Strata (n=600, all items labeled by all three annotators, blind):
  cord_sonnet_accepted 150 | cord_sonnet_general 150 (grounded x auto_correct)
  funsd_general 200        | cord_haiku 50 | cord_qwen 50

Outputs under --out (default annotation_package/):
  images/<doc_id>.png                copied doc images
  annotator_{A,B,C}.csv              per-annotator files (independent shuffles;
                                     NO auto labels / model names / strata)
  manifest.json                      hidden key: item -> stratum, auto label,
                                     grounded, accepted flag, source dump
  ANNOTATION_GUIDE.md                copied from docs/

Run on the machine that has the images (the GPU pod):
  python scripts/campaign2026/make_annotation_batch.py --out annotation_package
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 7
ALPHA = 0.10

DUMPS = {
    "cord_sonnet": "data/apivlm_perfield_rich_cord.json",
    "funsd": "data/apivlm_perfield_rich_funsd.json",
    "cord_haiku": "data/apivlm_perfield_rich_cord_claude-haiku-4-5.json",
    "cord_qwen": "data/apivlm_perfield_rich_cord_qwen14b.json",
}


def image_path(doc_id: str) -> Path | None:
    if doc_id.startswith("cord-"):
        # cord-train-00004 -> data/cord_images/train/00004.png
        _, split, idx = doc_id.split("-")
        return Path(f"data/cord_images/{split}/{idx}.png")
    if doc_id.startswith("funsd-"):
        # funsd-testing-82504862 -> data/funsd/dataset/testing_data/images/82504862.png
        _, split, stem = doc_id.split("-", 2)
        return Path(f"data/funsd/dataset/{split}_data/images/{stem}.png")
    return None


# ---- production acceptance (hgb_split_noent recipe, documented in the manifest) ----
def _leaf(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path).split(".")[-1]


def _features(recs, vocab):
    rows = []
    for r in recs:
        v = str(r.get("value") or "")
        verb = float(r.get("verbalized") or 0.0)
        cons = float(r.get("consistency") or 0.0)
        sup = float(r.get("support") or 0.0)
        grd = 1.0 if r.get("grounded") else 0.0
        digits = sum(c.isdigit() for c in v)
        feat = [
            verb, cons, sup, grd,
            np.log1p(len(v)), (digits / len(v)) if v else 0.0,
            1.0 if re.fullmatch(r"[0-9.,\- ]+", v or "_") else 0.0,
            1.0 if sup == 0.0 else 0.0, cons * verb,
        ]
        lf = _leaf(r["path"])
        feat.extend(1.0 if lf == w else 0.0 for w in vocab)
        rows.append(feat)
    return np.asarray(rows)


def production_accepted(recs) -> set[int]:
    """Indices of test fields accepted by the production config at ALPHA.

    Recipe: seed-7 doc split (half cal / half test); within cal, seed-7
    fit/val doc split; depth-3 HistGradientBoosting (no entailment feature,
    fieldtype one-hots with fit-half freq >= 25); add-one threshold on val.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(SEED)
    docs = np.array([r["doc_id"] for r in recs])
    y = np.array([1.0 if r.get("correct") in (True, 1) else 0.0 for r in recs])
    uniq = np.unique(docs)
    perm = rng.permutation(uniq)
    cal_docs = set(perm[: len(uniq) // 2].tolist())
    cal_mask = np.array([d in cal_docs for d in docs])
    cal_uniq = perm[: len(uniq) // 2]
    fit_docs = set(rng.permutation(cal_uniq)[: len(cal_uniq) // 2].tolist())
    fit_mask = np.array([d in fit_docs for d in docs]) & cal_mask
    val_mask = cal_mask & ~fit_mask

    fit_recs = [r for r, m in zip(recs, fit_mask) if m]
    from collections import Counter

    leaf_counts = Counter(_leaf(r["path"]) for r in fit_recs)
    vocab = sorted(w for w, c in leaf_counts.items() if c >= 25)
    X = _features(recs, vocab)
    clf = HistGradientBoostingClassifier(max_depth=3, random_state=SEED)
    clf.fit(X[fit_mask], y[fit_mask])
    s = clf.predict_proba(X)[:, 1]

    sv, yv = s[val_mask], y[val_mask]
    order = np.argsort(-sv)
    err = 0
    thr = None
    for k, i in enumerate(order, 1):
        err += 1 - yv[i]
        if (err + 1) / (k + 1) <= ALPHA:
            thr = sv[i]
    if thr is None:
        return set()
    test_mask = ~cal_mask
    return {i for i in np.where(test_mask & (s >= thr))[0].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="annotation_package")
    ap.add_argument("--per-annotator-shuffle-seeds", default="101,202,303")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = {k: json.loads(Path(v).read_text()) for k, v in DUMPS.items()}
    items: list[dict] = []

    def add(rec, source, stratum):
        items.append(
            {
                "item_id": f"i{len(items):04d}",
                "doc_id": rec["doc_id"],
                "path": rec["path"],
                "value": str(rec.get("value") or ""),
                "source": source,
                "stratum": stratum,
                "auto_correct": int(rec.get("correct") in (True, 1)),
                "grounded": int(bool(rec.get("grounded"))),
            }
        )

    def strat_sample(recs, n, source, stratum):
        """~n/4 from each grounded x auto_correct cell (fallback to available)."""
        cells: dict[tuple[int, int], list] = {}
        for r in recs:
            cells.setdefault(
                (int(bool(r.get("grounded"))), int(r.get("correct") in (True, 1))), []
            ).append(r)
        per = n // 4
        chosen = []
        for key in sorted(cells):
            pool = cells[key]
            take = min(per, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            chosen.extend(pool[i] for i in idx)
        # top up from the union if a cell was short
        short = n - len(chosen)
        if short > 0:
            seen = {id(r) for r in chosen}
            rest = [r for r in recs if id(r) not in seen]
            idx = rng.choice(len(rest), size=min(short, len(rest)), replace=False)
            chosen.extend(rest[i] for i in idx)
        for r in chosen:
            add(r, source, stratum)

    # 1) accepted set (production config) — human-verified selective risk
    acc_idx = sorted(production_accepted(data["cord_sonnet"]))
    take = rng.choice(len(acc_idx), size=min(150, len(acc_idx)), replace=False)
    for i in take:
        add(data["cord_sonnet"][acc_idx[i]], "cord_sonnet", "accepted@0.10")
    # 2) general strata
    strat_sample(data["cord_sonnet"], 150, "cord_sonnet", "general")
    strat_sample(data["funsd"], 200, "funsd", "general")
    strat_sample(data["cord_haiku"], 50, "cord_haiku", "general")
    strat_sample(data["cord_qwen"], 50, "cord_qwen", "general")

    # images
    missing = 0
    for it in items:
        src = image_path(it["doc_id"])
        dst = out / "images" / f"{it['doc_id']}.png"
        if src and src.exists():
            if not dst.exists():
                shutil.copy(src, dst)
        else:
            missing += 1
    # annotator CSVs (blind, independently shuffled)
    seeds = [int(x) for x in args.per_annotator_shuffle_seeds.split(",")]
    for name, sseed in zip("ABC", seeds):
        order = np.random.default_rng(sseed).permutation(len(items))
        with open(out / f"annotator_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "image", "field", "predicted_value", "label", "notes"])
            for i in order:
                it = items[i]
                w.writerow(
                    [it["item_id"], f"images/{it['doc_id']}.png", it["path"], it["value"], "", ""]
                )
    (out / "manifest.json").write_text(json.dumps({"seed": SEED, "alpha": ALPHA, "items": items}, indent=1))
    guide = Path("docs/ANNOTATION_GUIDE.md")
    if guide.exists():
        shutil.copy(guide, out / "ANNOTATION_GUIDE.md")
    n_acc = sum(1 for i in items if i["stratum"] == "accepted@0.10")
    print(
        f"package: {len(items)} items ({n_acc} accepted-set), "
        f"{len(set(i['doc_id'] for i in items))} docs, {missing} missing images -> {out}"
    )


if __name__ == "__main__":
    main()
