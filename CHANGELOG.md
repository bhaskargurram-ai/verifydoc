# Changelog

## v0.1.0 — 2026-07-21

First public release.

- **Core types**: `FieldPrediction` / `FieldGold` / `Grounding` / `Document` /
  executable `Schema` (per-leaf scoring rules), path flatten/unflatten.
- **Adapters**: pluggable `ExtractorAdapter` interface; mock (canned + seeded
  noisy), text-search baseline, PaddleOCR-VL, dots.ocr, Docling/MinerU output,
  API-VLM (injectable client).
- **Confidence signals**: token-prob, verbalized, k-sample consensus voting,
  grounding-based, weighted combined.
- **Calibrators**: temperature, Platt, isotonic, histogram + split-conformal
  abstention with finite-sample E[risk] ≤ α guarantee; split-disjointness
  guards (never tune on test, enforced in code).
- **Grounding stage**: value → page/bbox/char-span attachment with support.
- **Policy**: empirical + conformal target-risk thresholds; accept/review.
- **Eval harness (VerifyDocBench scorer)**: extraction (P/R/F1, exact,
  CER/WER, ANLS, TEDS/TEDS-Struct, GriTS Con/Top/Loc, omission vs
  hallucination), calibration (ECE 15-bin, Adaptive ECE, MCE, Brier, NLL,
  TCE, reliability), selective (RC, AURC, E-AURC, Coverage@Risk,
  Risk@Coverage, Acc@k, AUROC, AUPR, FPR@95%TPR), grounding (IoU acc @
  0.5/0.7, mean IoU, span F1, grounding-conditioned correctness), stats
  (bootstrap CIs, paired permutation/bootstrap tests). All numerically
  regression-tested against hand-computed fixtures.
- **Benchmark**: deterministic synthetic slice with gold values *and* gold
  boxes; CORD v2 loader; `make results` regenerates all tables/figures from
  pinned configs.
- **UX**: typer CLI (`verifydoc extract`), Streamlit review UI with
  click-through grounding.
