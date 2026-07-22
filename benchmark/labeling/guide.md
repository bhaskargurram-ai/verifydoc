# VerifyDocBench labeling guide (v0.1)

## Per-field correctness
For each predicted field, mark `correct` iff the value matches the document
under the field's scoring rule (exact / numeric-tolerance / semantic). Mark
`omission` when the document contains the field but the extraction lacks it;
mark `hallucination` when the extraction asserts a field the document does not
support. Omission ≠ hallucination — never conflate them.

## Gold source boxes
Draw the tightest axis-aligned box around the exact glyphs the value is read
from (not the label). Multi-line values: one box per line, union recorded.
Coordinates are normalized to page width/height.

## Agreement protocol
Every slice: two annotators label an overlapping 10% sample; report Cohen's κ
for correctness labels and mean IoU for boxes. Disagreements are adjudicated
by a third pass and the guide is amended with the ruling.

## Tooling
- Each annotator saves a JSON label file:
  `{"annotator": "alice", "labels": {"<field_id>": 0|1}}`.
- Compute agreement across files:
  `verifydoc iaa alice.json bob.json [carol.json ...]`
  → Fleiss' κ (all annotators), pairwise Cohen's κ, and a κ-band interpretation
  (Landis–Koch). Implemented in `verifydoc/labeling.py` and
  `verifydoc/eval/stats.py` (`cohens_kappa`, `fleiss_kappa`), unit-tested.

## Why human labels matter most for free text
`scripts/annotator_agreement.py` measures agreement between two *automatic*
scoring protocols (strict exact-match vs schema-typed) on real predictions as a
lower-bound proxy. It finds Cohen's κ ≈ 0.78 on CORD (structured/numeric labels
are robust to protocol) but κ ≈ 0.10 on FUNSD (free-text correctness is highly
protocol-dependent). This is exactly why human labeling with reported IAA is
prioritized for semantic/free-text fields.
