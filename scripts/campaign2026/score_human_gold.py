#!/usr/bin/env python3
"""Score the returned human-annotation CSVs (issue #68) into paper-ready numbers.

Inputs: the annotation package dir (with manifest.json) + the three filled CSVs.
Outputs (printed + JSON):
  1. IAA: Fleiss' kappa + pairwise Cohen's (overall, per dataset, per stratum)
     via verifydoc.labeling (also writes per-annotator JSON for `verifydoc iaa`).
  2. Human gold = 2-of-3 majority (items with >=2 'U' are excluded, reported).
  3. kappa(automatic label, human gold) per dataset — the label-reliability
     upgrade for Paper B (replaces the protocol-vs-protocol proxy).
  4. HUMAN-VERIFIED SELECTIVE RISK of the production accepted set
     (stratum accepted@0.10): 1 - mean(human gold), with an exact
     Clopper-Pearson 95% CI — the guarantee-validation upgrade for Paper A.
  5. Grounding gap on human gold (per dataset).

Usage:
  python scripts/campaign2026/score_human_gold.py --package annotation_package \
      --csvs annotator_A_filled.csv annotator_B_filled.csv annotator_C_filled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.stats import beta as beta_dist  # noqa: E402

from verifydoc.eval.stats import cohens_kappa  # noqa: E402
from verifydoc.labeling import iaa_report  # noqa: E402


def clopper_pearson(k: int, n: int, ci: float = 0.95):
    if n == 0:
        return (0.0, 1.0)
    a = (1 - ci) / 2
    lo = beta_dist.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def read_csv_labels(path: Path) -> dict[str, str]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().upper()
            if lab in {"0", "1", "U"}:
                out[row["item_id"]] = lab
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="annotation_package")
    ap.add_argument("--csvs", nargs=3, required=True, help="three filled annotator CSVs")
    ap.add_argument("--out", default="annotation_package/human_gold_report.json")
    args = ap.parse_args()

    manifest = json.loads((Path(args.package) / "manifest.json").read_text())
    items = {it["item_id"]: it for it in manifest["items"]}
    raw = {f"annot{chr(65 + i)}": read_csv_labels(Path(p)) for i, p in enumerate(args.csvs)}

    # 1) IAA on binary labels (drop U per annotator; iaa uses co-labeled overlap)
    binary = {a: {k: int(v) for k, v in labs.items() if v in {"0", "1"}} for a, labs in raw.items()}
    for a, labs in binary.items():
        Path(args.package, f"labels_{a}.json").write_text(
            json.dumps({"annotator": a, "labels": labs})
        )
    rep = iaa_report(binary)
    print("IAA overall:", rep.interpret())

    per_ds_iaa = {}
    for ds in sorted({it["source"] for it in items.values()}):
        sub = {
            a: {k: v for k, v in labs.items() if items[k]["source"] == ds}
            for a, labs in binary.items()
        }
        r = iaa_report(sub)
        per_ds_iaa[ds] = {"fleiss": r.fleiss, "n": r.n_items}
        print(f"  IAA {ds}: fleiss={r.fleiss:.3f} (n={r.n_items})")

    # 2) human gold: majority of available binary votes; need >=2 votes
    gold: dict[str, int] = {}
    excluded = 0
    for iid in items:
        votes = [binary[a][iid] for a in binary if iid in binary[a]]
        if len(votes) >= 2 and (votes.count(1) != votes.count(0)):
            gold[iid] = 1 if votes.count(1) > votes.count(0) else 0
        elif len(votes) == 3:  # 3 votes but tie impossible with 3; keep majority
            gold[iid] = 1 if votes.count(1) >= 2 else 0
        else:
            excluded += 1
    print(f"human gold: {len(gold)} items ({excluded} excluded: too many U / ties)")

    # 3) kappa(auto, human) per dataset
    kappa_auto = {}
    for ds in per_ds_iaa:
        ids = [i for i in gold if items[i]["source"] == ds]
        if len(ids) < 10:
            continue
        auto = [items[i]["auto_correct"] for i in ids]
        hum = [gold[i] for i in ids]
        k = cohens_kappa(auto, hum)
        agree = sum(a == h for a, h in zip(auto, hum)) / len(ids)
        kappa_auto[ds] = {"kappa": k, "raw_agreement": agree, "n": len(ids)}
        print(f"  kappa(auto,human) {ds}: {k:.3f} (agree {agree:.3f}, n={len(ids)})")

    # 4) human-verified selective risk of the accepted set
    acc_ids = [i for i in gold if items[i]["stratum"] == "accepted@0.10"]
    errs = sum(1 - gold[i] for i in acc_ids)
    lo, hi = clopper_pearson(errs, len(acc_ids))
    print(
        f"HUMAN-VERIFIED SELECTIVE RISK (accepted@0.10): {errs}/{len(acc_ids)} = "
        f"{errs / max(1, len(acc_ids)):.3f}  95% CI [{lo:.3f},{hi:.3f}]  "
        f"(alpha={manifest['alpha']})"
    )

    # 5) grounding gap on human gold
    gaps = {}
    for ds in per_ds_iaa:
        ids = [i for i in gold if items[i]["source"] == ds and items[i]["stratum"] != "accepted@0.10"]
        g1 = [gold[i] for i in ids if items[i]["grounded"] == 1]
        g0 = [gold[i] for i in ids if items[i]["grounded"] == 0]
        if len(g1) >= 5 and len(g0) >= 5:
            gap = sum(g1) / len(g1) - sum(g0) / len(g0)
            gaps[ds] = {"gap": gap, "n_grounded": len(g1), "n_ungrounded": len(g0)}
            print(f"  grounding gap on human gold {ds}: {gap:+.3f} ({len(g1)}/{len(g0)})")

    Path(args.out).write_text(
        json.dumps(
            {
                "iaa_overall": {"fleiss": rep.fleiss, "mean_cohen": rep.mean_pairwise_cohen,
                                "n_items": rep.n_items},
                "iaa_per_dataset": per_ds_iaa,
                "kappa_auto_vs_human": kappa_auto,
                "accepted_set_risk": {
                    "errors": errs, "n": len(acc_ids),
                    "risk": errs / max(1, len(acc_ids)), "ci95": [lo, hi],
                    "alpha": manifest["alpha"],
                },
                "grounding_gap_human_gold": gaps,
                "excluded_items": excluded,
            },
            indent=1,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
