#!/usr/bin/env python3
"""Build the XFUND human-annotation package (follow-up to issue #68): 400 items.

100 items per language (de/es/fr/zh), balanced 2x2 over grounded x auto_correct,
all items labeled by all three annotators, blind. Images are copied from the
XFUND v1.0 val zips (doc index i in the loader's order -> {lang}_val_{i}.jpg).

Run from the repo root after extracting the zips to --images-root:
  python scripts/campaign2026/make_annotation_batch_xfund.py \
      --images-root /path/to/xfund_images --out annotation_package_xfund
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 7
LANGS = ["de", "es", "fr", "zh"]
N_PER_LANG = 100
DUMPS = {
    "de": "data/apivlm_perfield_rich_xfund.json",
    "es": "data/apivlm_perfield_rich_xfund_es.json",
    "fr": "data/apivlm_perfield_rich_xfund_fr.json",
    "zh": "data/apivlm_perfield_rich_xfund_zh.json",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-root", required=True, help="dir with de/ es/ fr/ zh/ extracted val images")
    ap.add_argument("--out", default="annotation_package_xfund")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    items: list[dict] = []
    for lang in LANGS:
        recs = json.loads(Path(DUMPS[lang]).read_text())
        cells: dict[tuple[int, int], list] = {}
        for r in recs:
            cells.setdefault(
                (int(bool(r.get("grounded"))), int(r.get("correct") in (True, 1))), []
            ).append(r)
        per = N_PER_LANG // 4
        chosen = []
        for key in sorted(cells):
            pool = cells[key]
            take = min(per, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            chosen.extend(pool[i] for i in idx)
        short = N_PER_LANG - len(chosen)
        if short > 0:
            seen = {id(r) for r in chosen}
            rest = [r for r in recs if id(r) not in seen]
            idx = rng.choice(len(rest), size=min(short, len(rest)), replace=False)
            chosen.extend(rest[i] for i in idx)
        for r in chosen:
            items.append(
                {
                    "item_id": f"i{len(items):04d}",
                    "doc_id": r["doc_id"],
                    "path": r["path"],
                    "value": str(r.get("value") or ""),
                    "source": f"xfund_{lang}",
                    "stratum": "general",
                    "auto_correct": int(r.get("correct") in (True, 1)),
                    "grounded": int(bool(r.get("grounded"))),
                }
            )

    missing = 0
    root = Path(args.images_root)
    for it in items:
        lang = it["source"].split("_")[1]
        idx = int(it["doc_id"].rsplit("-", 1)[1])
        src = root / lang / f"{lang}_val_{idx}.jpg"
        dst = out / "images" / f"{it['doc_id']}.jpg"
        if src.exists():
            if not dst.exists():
                shutil.copy(src, dst)
        else:
            missing += 1

    for name, sseed in zip("ABC", (101, 202, 303)):
        order = np.random.default_rng(sseed).permutation(len(items))
        with open(out / f"annotator_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "image", "field", "predicted_value", "label", "notes"])
            for i in order:
                it = items[i]
                w.writerow(
                    [it["item_id"], f"images/{it['doc_id']}.jpg", it["path"], it["value"], "", ""]
                )
    (out / "manifest.json").write_text(json.dumps({"seed": SEED, "items": items}, indent=1))
    guide = Path("docs/ANNOTATION_GUIDE.md")
    if guide.exists():
        shutil.copy(guide, out / "ANNOTATION_GUIDE.md")
    per_lang = Counter(i["source"] for i in items)
    print(
        f"package: {len(items)} items {dict(per_lang)}, "
        f"{len(set(i['doc_id'] for i in items))} docs, {missing} missing images -> {out}"
    )


if __name__ == "__main__":
    main()
