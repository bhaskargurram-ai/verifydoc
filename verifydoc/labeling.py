"""Human-labeling reliability: aggregate multiple annotators into IAA.

VerifyDocBench derives correctness labels automatically where gold values
exist; for fields that need human judgement (free-text equivalence, ambiguous
grounding), a full release collects labels from >=2 annotators and reports
inter-annotator agreement. This module turns per-annotator label files into a
Cohen's/Fleiss' kappa report; ``verifydoc iaa <files...>`` is the CLI over it.

A label file is JSON: ``{"annotator": "alice", "labels": {"<field_id>": 0|1}}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from verifydoc.eval.stats import cohens_kappa, fleiss_kappa


@dataclass
class IAAReport:
    n_items: int
    n_annotators: int
    pairwise_cohen: dict[tuple[str, str], float] = field(default_factory=dict)
    fleiss: float = 0.0

    @property
    def mean_pairwise_cohen(self) -> float:
        vals = list(self.pairwise_cohen.values())
        return sum(vals) / len(vals) if vals else 0.0

    def interpret(self) -> str:
        k = self.fleiss
        band = (
            "poor"
            if k < 0.0
            else (
                "slight"
                if k < 0.20
                else (
                    "fair"
                    if k < 0.40
                    else "moderate" if k < 0.60 else "substantial" if k < 0.80 else "almost perfect"
                )
            )
        )
        return (
            f"{self.n_annotators} annotators, {self.n_items} co-labeled items; "
            f"Fleiss' kappa = {k:.3f} ({band}); "
            f"mean pairwise Cohen's kappa = {self.mean_pairwise_cohen:.3f}"
        )


def iaa_report(annotations: dict[str, dict[str, int]]) -> IAAReport:
    """Compute pairwise Cohen's + Fleiss' kappa over annotators.

    ``annotations`` maps annotator name -> {field_id: label}. Only the field
    ids labeled by *every* annotator are scored (the co-labeled overlap).
    """
    if len(annotations) < 2:
        raise ValueError("need at least two annotators")
    names = sorted(annotations)
    shared = set.intersection(*(set(annotations[n]) for n in names))
    if not shared:
        raise ValueError("annotators share no co-labeled items")
    items = sorted(shared)

    pairwise = {
        (a, b): cohens_kappa([annotations[a][i] for i in items], [annotations[b][i] for i in items])
        for a, b in combinations(names, 2)
    }

    categories = sorted({annotations[n][i] for n in names for i in items})
    cat_index = {c: k for k, c in enumerate(categories)}
    counts = []
    for i in items:
        row = [0] * len(categories)
        for n in names:
            row[cat_index[annotations[n][i]]] += 1
        counts.append(row)
    fleiss = fleiss_kappa(counts) if len(categories) > 1 else 1.0

    return IAAReport(
        n_items=len(items), n_annotators=len(names), pairwise_cohen=pairwise, fleiss=fleiss
    )


def load_annotations(paths: list[str | Path]) -> dict[str, dict[str, int]]:
    """Load annotator label files (JSON: {annotator, labels})."""
    out: dict[str, dict[str, int]] = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        name = data.get("annotator") or Path(p).stem
        out[name] = {str(k): int(v) for k, v in data["labels"].items()}
    return out
