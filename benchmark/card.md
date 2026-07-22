# VerifyDocBench — dataset card (v0.1)

**What it is:** documents + JSON schemas + gold values **plus per-field
correctness labels and gold source boxes**, with a harness that scores
extraction quality, calibration, selective risk, and grounding
(`verifydoc/eval/`). The harness scores anything that emits `FieldPrediction`.

## Slices

| Slice | Source | License | Status | Gold boxes |
|---|---|---|---|---|
| `synthetic` | generated in-repo (`benchmark/datasets/synthetic.py`) | Apache-2.0 | shipped, CI-run | yes (from layout) |
| `cord` | naver-clova-ix/cord-v2 (HF hub) | CC-BY-4.0 (verify at release) | shipped — real text layers, harness-run | yes (located word quads) |
| `funsd` | guillaumejaume.github.io/FUNSD | research use | shipped — real forms, harness-run | yes (annotated answer boxes) |
| `sroie` | darentang/sroie (HF) | ICDAR 2019 research use | shipped — text layer + located gold boxes | yes (located) |
| `docile` / `xfund` | official downloads | varies — some research-only | planned | planned |

## Redistribution policy (PROJECT.md §5.H)

We redistribute **our added annotations** (per-field correctness labels, gold
source boxes, schemas, splits, scripts) and *reference* the original
downloads. Restrictively licensed source images are never re-hosted. Each
slice's license is stated here before release.

## Splits

Every slice ships with `train` / `calibration` / `test` doc-id lists.
Calibration and abstention thresholds are fit **only** on `calibration`;
`verifydoc.calibration.assert_disjoint` is called by the harness on every run.

## Labeling

Where correctness/boxes are not derivable automatically, fields are labeled by
≥2 annotators on a sample with Cohen's κ reported (`benchmark/labeling/` holds
the guide + IAA scripts). The synthetic and CORD slices carry automatic
correctness (gold parses exist), per PROJECT.md §12.
