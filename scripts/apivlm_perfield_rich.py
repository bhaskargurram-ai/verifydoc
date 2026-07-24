#!/usr/bin/env python
"""Phase-1 rich per-field signal capture from a real frontier VLM.

Extends apivlm_perfield.py: for each field it persists *all* the signals the
grounding-conditioned method can fuse, so the method can be developed and
evaluated on genuine model outputs (not a simulation):

  value, verbalized, consistency (k-sample consensus agreement),
  grounded, support (AMBIGUITY-PENALIZED), entailment (NLI: does the grounded
  span entail "field = value"?), correct.

Outputs data/apivlm_perfield_rich.json. Env: ANTHROPIC_API_KEY.
Usage: python scripts/apivlm_perfield_rich.py --limit 400 --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.adapters.api_vlm import APIVLMAdapter  # noqa: E402
from verifydoc.confidence import consensus, verbalized_confidence  # noqa: E402
from verifydoc.confidence.entailment import _hypothesis, _premise  # noqa: E402
from verifydoc.eval.extraction import score_fields  # noqa: E402
from verifydoc.grounding import ground_predictions  # noqa: E402

CACHE = Path("data/apivlm_perfield_rich.json")


def _page_text(doc) -> str:
    return "\n".join(p.text or "" for p in doc.pages if p.text)


def one_doc(adapter, item, k):
    """Return per-field records (entailment filled in a later batched pass)."""
    samples = adapter.extract_samples(item.doc, item.schema, k)  # k temp>0 runs
    cons = consensus(samples)  # modal value + agreement (=consistency) per path
    grounded = ground_predictions(cons, item.doc)  # ambiguity-penalized support
    report = score_fields(grounded, item.golds)
    by_path = {p.path: p for p in grounded}
    src = _page_text(item.doc)
    out = []
    for fs in report.field_scores:
        p = by_path[fs.path]
        verb = verbalized_confidence(p)
        prem = _premise(p, src) if p.grounding is not None else None
        rec = {
            "doc_id": item.doc.doc_id,
            "path": fs.path,
            "value": "" if p.value is None else str(p.value),
            "verbalized": float(verb) if verb is not None else 0.9,
            "consistency": float(p.confidence),  # consensus agreement over k samples
            "grounded": p.grounding is not None,
            "support": float(p.grounding.support) if p.grounding else 0.0,
            "correct": int(fs.correct),
            "_premise": prem,
            "_hyp": _hypothesis(p, "{field} is {value}") if prem is not None else None,
        }
        out.append(rec)
    return out


def add_entailment(records):
    """Batch the NLI cross-encoder over grounded fields (GPU) and fill entailment."""
    pairs, idx = [], []
    for i, r in enumerate(records):
        if r.get("_premise") and r.get("_hyp"):
            pairs.append((r["_premise"], r["_hyp"]))
            idx.append(i)
    for r in records:
        r["entailment"] = 0.0
    if not pairs:
        return records
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        import numpy as np

        logits = np.asarray(model.predict(pairs))  # [n,3] contradiction/entail/neutral
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        for j, i in enumerate(idx):
            records[i]["entailment"] = float(probs[j][1])  # entailment class
    except Exception as exc:  # NLI optional — keep the run if it fails
        print(f"[entailment skipped: {exc}]", flush=True)
    for r in records:
        r.pop("_premise", None)
        r.pop("_hyp", None)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from benchmark.datasets import cord

    bench = cord.load(split=args.split, limit=args.limit, with_images=True)
    adapter = APIVLMAdapter()  # default claude-sonnet-5, reads ANTHROPIC_API_KEY
    records: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_doc, adapter, item, args.k): item for item in bench}
        for fut in as_completed(futs):
            try:
                records.extend(fut.result())
            except Exception as exc:
                print(f"[doc failed: {exc}]", flush=True)
            done += 1
            if done % 25 == 0:
                print(f"...{done}/{len(bench)} docs, {len(records)} fields", flush=True)

    print(
        f"extraction done: {len(records)} fields from {len(bench)} docs; scoring NLI...", flush=True
    )
    records = add_entailment(records)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(records, indent=0), encoding="utf-8")
    corr = sum(r["correct"] for r in records) / max(1, len(records))
    grnd = sum(r["grounded"] for r in records) / max(1, len(records))
    print(f"wrote {CACHE}: {len(records)} fields, correct={corr:.3f}, grounded={grnd:.3f}")


if __name__ == "__main__":
    main()
