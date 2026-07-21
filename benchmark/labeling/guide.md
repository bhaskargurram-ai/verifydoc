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
