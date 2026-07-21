# USP validation — what VerifyDoc has that others don't

*Self-audit performed 2026-07-21 while building v0.1.0; re-verify before the
paper submission.*

## The claim

**VerifyDoc is the only open-source, model-agnostic layer that returns, for
every extracted field: a calibrated confidence, a source grounding, and an
accept/review decision with a distribution-free risk guarantee — plus the
benchmark harness that proves whether those numbers mean anything.**

## The evidence

### 1. The popular OSS parsers do not ship this contract
Docling, MinerU, Marker, olmOCR, DeepSeek-OCR, PaddleOCR-VL parse documents
into Markdown/JSON. Their outputs carry at most raw OCR recognition scores —
no field-level calibrated probability, no abstention policy, no
calibration/selective-risk evaluation. (Surveyed guides: MarkTechPost
"Structured PDF-to-JSON 2026", Jimmy Song "PDF-to-Markdown deep dive",
Spheron self-hosting guide.)

### 2. Commercial APIs prove the demand — and keep it closed
Box shipped **field-level confidence via API in Jan 2026**; Azure Document
Intelligence, AWS Textract, Extend, and Iteration Layer all sell per-field
confidence + human-in-the-loop routing. The reliability contract is a paid
feature of closed platforms. VerifyDoc is the open implementation.

### 3. The research gap is explicitly named
- ExtractBench (arXiv:2602.12247) names "confidence calibration metrics —
  measuring whether models know when they're uncertain" as future work.
- CRC-for-structured-generation (arXiv:2606.29054) shows raw model confidence
  has ECE up to 0.61 and that certification forces abstention — but ships no
  reusable tool.
- No benchmark evaluates per-field calibration/abstention/grounding for
  document extraction. VerifyDocBench is the first.

### 4. The name is free
No `verifydoc` package exists on PyPI (checked 2026-07-21: 404) and no
same-named Python library on GitHub.

## Feature comparison

| Capability | VerifyDoc | Docling / MinerU / Marker | PaddleOCR-VL / dots.ocr | Box / Azure / Textract |
|---|---|---|---|---|
| Parse to structured output | via any of them (adapters) | ✅ | ✅ | ✅ |
| Per-field **calibrated** confidence | ✅ (5 calibrators, ECE-verified) | ❌ | raw OCR scores only | ✅ closed |
| Accept/review abstention at target risk | ✅ (+ conformal guarantee) | ❌ | ❌ | partial, closed |
| Per-field grounding (page/bbox/span) | ✅ | layout only, not per-field | boxes, not per-field values | partial, closed |
| Calibration + selective-risk benchmark | ✅ (VerifyDocBench) | ❌ | ❌ | ❌ |
| Open source | ✅ Apache-2.0 | ✅ | ✅ | ❌ |

## Built-in self-checks (run in CI, not just claimed)

- `tests/test_harness.py::test_signals_carry_real_signal` — informative
  signals (consensus/grounding/combined) must out-rank the deliberately
  overconfident verbalized baseline on Coverage@risk.
- `tests/test_harness.py::test_grounding_gap_positive` — the grounded-fields-
  are-more-often-correct hypothesis must hold on the benchmark slice.
- `tests/test_calibrators.py::test_risk_guarantee_holds_empirically` — the
  conformal abstention guarantee is checked over 200 simulated cal/test
  splits, not assumed.

## Sources

- https://support.box.com/hc/en-us/articles/48546593181459-Confidence-Scores-via-API-in-Box-Extract-Jan-2026
- https://www.extend.ai/resources/best-confidence-scoring-systems-document-processing
- https://iterationlayer.com/blog/ai-data-extraction-confidence-scores
- https://www.marktechpost.com/2026/07/04/structured-pdf-to-json-a-guide-to-open-source-extraction-models-in-2026/
- https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/
- https://arxiv.org/abs/2602.12247 (ExtractBench)
- https://arxiv.org/abs/2606.29054 (CRC for structured generation)
- https://arxiv.org/abs/2409.04117 (confidence-aware OCR error detection)
