#!/usr/bin/env python
"""Backfill the entailment-NLI signal into an existing rich dump (no API calls).

The capture's NLI pass failed under a transformers/torch mismatch, leaving
entailment=0. This re-grounds each cached value to recover its source span and
scores "field = value" entailment with an MNLI cross-encoder, merging the result
back into the dump. Reuses the cached VLM extraction; only re-grounds + runs NLI.

Usage: python scripts/backfill_entailment.py --dump data/apivlm_perfield_rich.json \
    --split train --limit 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.confidence.entailment import _hypothesis, _premise  # noqa: E402
from verifydoc.grounding.attach import MIN_SUPPORT, _locate  # noqa: E402
from verifydoc.types import FieldPrediction  # noqa: E402


def _page_text(doc) -> str:
    return "\n".join(p.text or "" for p in doc.pages if p.text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="data/apivlm_perfield_rich.json")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    from benchmark.datasets import cord

    recs = json.loads(Path(args.dump).read_text())
    bench = cord.load(split=args.split, limit=args.limit, with_images=True)
    doc_by_id = {it.doc.doc_id: it.doc for it in bench}

    pairs, idx = [], []
    for i, r in enumerate(recs):
        r["entailment"] = 0.0
        if not r.get("grounded"):
            continue
        doc = doc_by_id.get(r["doc_id"])
        if doc is None:
            continue
        g = _locate(str(r["value"]), doc, MIN_SUPPORT, "uniform")
        if g is None or g.char_span is None:
            continue
        p = FieldPrediction(path=r["path"], value=r["value"], grounding=g)
        prem = _premise(p, _page_text(doc))
        if prem:
            pairs.append((prem, _hypothesis(p, "{field} is {value}")))
            idx.append(i)

    print(f"grounded fields with a span to score: {len(idx)}", flush=True)
    import numpy as np
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    logits = np.asarray(model.predict(pairs))
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    for j, i in enumerate(idx):
        recs[i]["entailment"] = float(probs[j][1])

    Path(args.dump).write_text(json.dumps(recs, indent=0), encoding="utf-8")
    nz = sum(r["entailment"] > 0 for r in recs) / max(1, len(recs))
    mean_g = (
        np.mean([r["entailment"] for r in recs if r.get("grounded")])
        if any(r.get("grounded") for r in recs)
        else 0.0
    )
    print(f"BACKFILL_DONE entailment nonzero={nz:.3f} mean_grounded={mean_g:.3f}")


if __name__ == "__main__":
    main()
