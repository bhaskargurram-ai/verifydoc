# Related work & positioning (2024–2026 literature sweep)

A structured survey of where VerifyDoc sits. Sources are cited in
`paper/refs.bib`; this is the engineer-facing summary.

## The gap, in one table

| Capability | OSS extraction libs (Instructor, Outlines, LangChain) | OSS parsers (Docling, Unstructured) | Confidence research (Beyond Logprobs) | Cleanlab TLM | Commercial IDP (Azure, Textract, Extend) | **VerifyDoc** |
|---|---|---|---|---|---|---|
| Schema-typed output | ✅ | — | ✅ | ✅ | ✅ | ✅ |
| Per-field **calibrated** confidence | ❌ | ❌ | ✅ | ✅ (uncalibrated to error rate) | ⚠️ opaque | ✅ |
| Source **grounding** (page/bbox/span) | ❌ | ✅ (parse-level) | ❌ | ❌ | ✅ | ✅ per-field |
| **Abstention** with a risk guarantee | ❌ | ❌ | ⚠️ threshold | ❌ | ⚠️ vendor thresholds | ✅ conformal |
| Open-source & self-hostable | ✅ | ✅ | (paper) | ❌ paid API | ❌ | ✅ |

**No system unifies calibrated per-field confidence + source grounding +
risk-controlled abstention, open-source.** That intersection is VerifyDoc.

## Benchmarks
Parsing accuracy is **saturated** (OmniDocBench ~94–96%) but real PDF→JSON
correctness collapses (ExtractBench ~4.6% field pass rate). ExtractBench,
VAREX, KIEval, OmniDocBench all measure accuracy; **none scores per-field
confidence, calibration, or abstention** — the axis VerifyDocBench adds.

## Confidence signals
- Closest competitor **Beyond Logprobs** (arXiv:2606.24420): multi-signal
  confidence for document field extraction (logprobs + consistency), ECE/AUROC/
  selective-risk — but **no grounding**.
- **Cleanlab TLM**: model-agnostic per-field trust scores — but closed, **no
  grounding, no abstention guarantee**.
- Semantic entropy (Nature 2024) & consistency/verbalized fusion (BSDetector)
  are SOTA wrong-answer detectors; both miss **self-consistent errors**, which
  is why an external verification signal (grounding) matters.

## Calibration & selective prediction
- Report **SmoothECE** (ICLR 2024) beside binned/adaptive ECE, and **AUGRC**
  (NeurIPS 2024) beside E-AURC — both now implemented in `eval/`.

## Conformal (the novel method's lineage)
- Conformal risk control (Angelopoulos et al., ICLR 2024), conformal factuality
  (Mohri & Hashimoto, ICML 2024), selective CRC (2025) all control **marginal**
  risk. Conditional-coverage theory (Gibbs–Cherian–Candès, JRSS-B 2025) shows
  exact per-field conditioning is impossible but exact **group-conditional**
  control is attainable (Mondrian CP, Vovk 2003).
- The closest foil, **CRC-certify** (arXiv:2606.29054), defines field-level
  JSON losses and a minimum-abstention bound but **explicitly does not condition
  on provenance**. VerifyDoc's **grounding-conditioned Mondrian CRC** fills
  exactly that gap — the first use of provenance as the conditioning taxonomy
  for conformal risk control in document extraction.

## Grounding / provenance
- Risk-controlled generative OCR (2603.19790), VISA (2412.14457), BoundingDocs
  (2501.03403) establish visual attribution and its IoU/span evaluation, and
  **document the short-value matching ambiguity** VerifyDoc addresses with
  **ambiguity-penalized support** (discount by number of equally-good matches).

## What this sweep changed in the code
- Ambiguity-penalized grounding support (real-data method now works: CORD
  0.01→0.72 coverage at 10% risk).
- AUGRC + SmoothECE metrics.
- Dynamic schemas (`to_json_schema`, `from_pydantic`) + `verify_model`.
