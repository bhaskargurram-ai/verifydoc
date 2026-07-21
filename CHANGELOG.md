# Changelog

## v0.2.0 — 2026-07-21

Real data, faster grounding, learned fusion.

- **CORD v2 slice**: real receipts with real text layers (word quads) and
  located gold boxes; Indonesian thousands-comma prices score numerically.
- **FUNSD slice**: 199 scanned forms; question→answer links become gold
  fields with exact annotated answer boxes.
- **Harness**: `dataset:` (synthetic | cord | funsd) and `extractor:` (any
  adapter registry name) dispatch; `make results` regenerates all slices.
- **Learned combiner**: logistic fusion over signal values + missing
  indicators, fit on the calibration split; reported as a `learned` row in
  every table (ablation vs the transparent weighted mean).
- **Grounder performance (~1000×)**: exact-match fast path, token-anchored
  fuzzy scan with length pruning, token-overlap scoring for paragraph-length
  values. FUNSD run: hours → 8 seconds.
- **Fix**: text-search label matching uses word boundaries ("total" no longer
  fires inside "Subtotal") — caught by the harness itself.
- Demo GIF rendered from a real pipeline run; GPU runbook
  (docs/REAL_MODELS.md) + pinned real-model config; paper skeleton.

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
