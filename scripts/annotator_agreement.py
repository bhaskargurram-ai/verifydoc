#!/usr/bin/env python
"""Labeling-reliability study: agreement between two scoring protocols.

VerifyDocBench's correctness labels are derived automatically from gold values.
A fair question for a benchmark is how *stable* those labels are to the scoring
protocol. We score the same real predictions two ways --- a strict
exact-match protocol and the schema-typed protocol (numeric tolerance +
semantic equivalence) --- and report Cohen's kappa between the two label sets.

This is a protocol-vs-protocol agreement (automatic), an honest lower-bound
proxy for the human inter-annotator agreement a full release would add; the
human-labeling tooling (`verifydoc.labeling`) and guide are shipped alongside.

Writes results/annotator_agreement.md. Offline; deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.adapters.mock import MockAdapter  # noqa: E402
from verifydoc.eval.extraction import value_correct  # noqa: E402
from verifydoc.eval.stats import cohens_kappa  # noqa: E402
from verifydoc.types import FieldGold  # noqa: E402

SEED = 7


def _label_sets(bench):
    """Two correctness labelings of the same predictions: exact vs schema-typed."""
    adapter = MockAdapter(
        gold={item.doc.doc_id: item.golds for item in bench},
        error_rate=0.25,
        omit_rate=0.05,
        hallucinate_rate=0.05,
        seed=SEED,
    )
    exact_labels, typed_labels = [], []
    for item in bench:
        gold_by_path = {g.path: g for g in item.golds}
        for pred in adapter.extract(item.doc, item.schema):
            gold = gold_by_path.get(pred.path)
            if gold is None or pred.value is None:
                continue
            typed_labels.append(int(value_correct(pred.value, gold)))
            strict = FieldGold(path=gold.path, value=gold.value, scoring="exact")
            exact_labels.append(int(value_correct(pred.value, strict)))
    return exact_labels, typed_labels


def main() -> None:
    from benchmark.datasets import cord, funsd

    rows = []
    for name, bench in [
        ("cord", cord.load(split="validation", limit=100)),
        ("funsd", funsd.load(split="testing")),
    ]:
        exact, typed = _label_sets(bench)
        kappa = cohens_kappa(exact, typed)
        agree = sum(a == b for a, b in zip(exact, typed)) / len(exact)
        rows.append(
            {"dataset": name, "n_fields": len(exact), "raw_agreement": agree, "cohens_kappa": kappa}
        )

    out = Path("results/annotator_agreement.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "n_fields", "raw_agreement", "cohens_kappa"]
    lines = [
        "# Labeling reliability: exact vs schema-typed scoring (Cohen's kappa)",
        "",
        "Agreement between two automatic correctness protocols on the same real",
        "predictions. High kappa = labels are robust to the scoring protocol.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['n_fields']} | "
            f"{r['raw_agreement']:.4f} | {r['cohens_kappa']:.4f} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in rows:
        print(
            f"{r['dataset']:6s}  n={r['n_fields']:4d}  raw_agreement={r['raw_agreement']:.3f}  "
            f"cohen_kappa={r['cohens_kappa']:.3f}"
        )


if __name__ == "__main__":
    main()
