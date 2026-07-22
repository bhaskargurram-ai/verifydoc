# VerifyDoc — A Trust Layer for Document → Structured-JSON Extraction

**Project document / build spec (v1.0).** This is the single source of truth for building VerifyDoc . It contains the motivation, the precise problem definition, the literature-grounded evaluation methodology (the core of the paper), the system architecture, the repository layout, a six-week implementation plan, the git workflow, and the paper plan. Hand this file (plus `CLAUDE.md`)  and build module by module against the "Definition of Done" in each section.

One-liner: *VerifyDoc wraps any document-extraction model and returns, for every field, a calibrated confidence score, a source-grounding box/span, and an abstain-or-accept decision — so a human reviews the 5% of fields that are actually wrong instead of eyeballing all of them.*

---

## 1. Motivation & positioning

Document → structured-JSON extraction is one of the highest-volume uses of LLMs/VLMs in production: invoices, receipts, forms, contracts, filings, lab reports. The open-source parsing stack that people actually run is huge and mature — MinerU, Docling, Marker, PaddleOCR-VL, dots.ocr, DeepSeek-OCR — and headline accuracy on the dominant benchmark (OmniDocBench, CVPR 2025) is now saturated: PaddleOCR-VL-1.6 reports 96.33% overall and GLM-OCR reports 94.6% on v1.5, beating frontier APIs. Yet the failure mode has *shifted*. The problem is no longer "the model can't read the page"; it is that modern VLMs produce **fluent, plausible, silently-wrong values** ($42.50 → $45.20) and expose **no reliable per-field signal telling you which values to trust**.

That is the wedge. VerifyDoc does not compete with the parsers — it *layers on top of any of them* and adds the missing reliability contract: field-level confidence + provenance + abstention. Commercial APIs already prove the demand (Box shipped field-level confidence in Jan 2026; Azure Document Intelligence, AWS Textract, and Extend all expose per-field confidence), but no popular open-source parser leads with it. Practitioner leaderboards name exactly this unmet need: whether a reviewer can trace an uncertain value back to the page before it enters a downstream system. VerifyDoc is the open-source answer.

Why the scope works: the tool is model-agnostic and thin (it wraps existing extractors), the paper contribution is the reliable "benchmark + released dataset + strong open baseline" type rather than a risky "beat-SOTA architecture" play, and the demo (highlight-the-hallucinated-field on the rendered page) is instantly shareable.

---

## 2. Problem statement (precise task)

**Input:** a document `D` (PDF or image, one or more pages) and a target JSON schema `S` (field names, types, nesting, arrays).

**Output:** a JSON object `J` conforming to `S`, where **every leaf field `f`** additionally carries:
- `value` — the extracted value,
- `confidence ∈ [0,1]` — a calibrated probability that `value` is correct,
- `grounding` — a source pointer: `{page, bbox}` (pixel region) and/or `{char_span}` (text-layer span) the value was read from, and
- `decision ∈ {accept, review}` — an abstention flag from a policy tuned to a target error rate.

**Success contract (the product promise):** at a chosen operating point, VerifyDoc auto-accepts as large a fraction of fields as possible (**coverage**) while holding the error rate among accepted fields (**selective risk**) below a user-set target (e.g., ≤ 2%). Everything below the confidence threshold is routed to `review` with its grounding attached so a human can verify it in seconds.

This reframes extraction as a **selective-prediction + calibration + grounding** problem layered on top of raw extraction — which is precisely the axis current benchmarks do *not* measure.

---

## 3. Literature review outcomes (what exists, what's missing)

### 3.1 Extraction / structured-output benchmarks
- **ExtractBench** (Ferguson et al., arXiv:2602.12247; ACM SIGKDD 2026; code: github.com/ContextualAI/extract-bench) is the closest prior work: 35 PDFs paired with JSON Schemas and human-annotated gold labels totalling 12,867 evaluatable fields. Its key methodological idea is treating **the schema as an executable specification** — each field declares its own scoring metric (exact-match for identifiers, numeric tolerance for quantities, semantic equivalence for names), arrays require alignment, and **omission is scored differently from hallucination**. Its headline finding: frontier models (GPT-5/5.2, Gemini-3, Claude 4.5) reach only ~4.6% field-level pass rate on realistic schemas and ~0% valid output on a 369-field enterprise schema. **Crucially, the authors name "confidence calibration metrics — measuring whether models know when they're uncertain" as explicit future work.** That sentence is the opening VerifyDoc walks through.
- **LLMStructBench** (arXiv:2602.14743): 22 models × 5 prompting strategies for text→JSON; finds prompting strategy matters more than model size and that structural validity ≠ semantic correctness.
- **Structured Output Benchmark / SOB** (arXiv:2604.25359): multi-source structured output; adds a 7-metric field-by-field pipeline with a parse→schema-compliance→semantic-scoring hardening rule.
- **JSONSchemaBench / StructEval**: schema-compliance / format-adherence only (necessary but not sufficient).
- **OCR/parsing benchmarks:** OmniDocBench (CVPR 2025; now considered saturated), OCRBench / OCRBench v2, READoc (ACL 2025). These score *reading/parsing quality*, not per-field confidence or abstention.

**Gap #1:** No benchmark evaluates whether a document-extraction system's **per-field confidence is calibrated** or whether its **abstention decisions** control real error. This is VerifyDoc's benchmark contribution.

### 3.2 Calibration & selective-prediction methodology (mature elsewhere, unused here)
Calibration (ECE/MCE/Brier/reliability diagrams) and selective prediction (risk–coverage, AURC/E-AURC, coverage@risk) are standard in classification and increasingly in LLM QA (e.g., selective-prediction and hallucination-detection papers in 2025–2026), but are essentially **absent from the document-extraction literature**. Recent LLM work also establishes the candidate confidence signals VerifyDoc will compare:
- **token/sequence probability** (log-probs) — requires logit access;
- **verbalized confidence** (ask the model to rate itself) — recent evidence suggests it can reflect richer answer-quality signals than token probability alone, but has systematic gaps and can be inflated by RLHF;
- **self-consistency / consensus** across multiple samples or multiple extractors — black-box, no logits needed;
- **conformal prediction / conformal risk control (CRC)** for distribution-free guarantees — recent structured-generation work (arXiv:2606.29054) shows CRC *can* certify structured outputs but that meaningful certification often forces heavy abstention, and that raw model confidence is a poor proxy (high ECE); conformal factuality/abstention (Mohri & Hashimoto 2024; Abbasi-Yadkori et al. 2024) are the reference methods.

**Gap #2:** Nobody has systematically compared these signals *for document field extraction* and turned the best one into a calibrated, abstaining, grounded tool. This is VerifyDoc's method + baseline contribution.

### 3.3 Grounding / verifiability
"Risk-Controlled Generative OCR" (arXiv:2603.19790) frames the exact risk: autoregressive decoding favors semantic plausibility while OCR requires outputs that are **visually grounded and geometrically verifiable**, creating deployment risk even at high benchmark accuracy. Grounded-mode OCR models (e.g., bbox-emitting variants) exist, which VerifyDoc can exploit to attach provenance.

**Net:** the topic is hot, the metrics exist in adjacent fields, the parsers are commoditized, and the reliability layer is validated by commercial APIs but unoccupied in open source. VerifyDoc = (tool) + (VerifyDocBench) + (strong open baseline) sitting squarely in that gap.

---

## 4. The contribution (three deliverables)

1. **VerifyDoc (the library):** `pip install verifydoc`; wraps any extractor (adapters for PaddleOCR-VL, dots.ocr, Docling/MinerU output, and any API VLM); emits per-field `confidence`, `grounding`, `decision`; ships a CLI, a Python API, and a Streamlit review UI that renders the page and highlights fields green/yellow/red with click-through to the source box.
2. **VerifyDocBench (the benchmark/dataset):** documents + schemas + gold values **plus per-field correctness labels and gold source boxes**, with an evaluation harness computing extraction quality, calibration, selective risk, and grounding. Built by extending public datasets (see §5.8).
3. **The paper:** a benchmark-and-empirical-study paper reporting the first systematic comparison of confidence signals + calibration + abstention for document extraction, with a strong, reproducible open baseline.

---

## 5. Performance metrics & evaluation (paper core)

This section is written so it can be lifted almost verbatim into the paper's "Evaluation" section and implemented directly in `verifydoc/eval/`. Metrics are grouped into four families — extraction quality (A), calibration (B), selective prediction / abstention (C), grounding (D) — followed by a SOTA reference table (E), the recommended primary/secondary suite (F), the expected baselines/ablations (G), and the experimental protocol (H).

Notation: a document yields a set of leaf fields; for field *i*, `ŷ_i` is the predicted value, `y_i` the gold value, `c_i ∈ [0,1]` the predicted confidence, and `correct_i ∈ {0,1}` an indicator that `ŷ_i` matches `y_i` under that field's scoring rule.

### 5.A Extraction-quality metrics (is the value right?)

| Metric | Definition (in words) | When to use | Range / direction |
|---|---|---|---|
| **Field P / R / F1** (entity-level) | Precision = correct predicted fields / all predicted; Recall = correct / all gold; F1 = harmonic mean. Compute at the leaf level after path-flattening nested JSON. | Primary headline for KIE (FUNSD, CORD, SROIE, DocILE, XFUND). | [0,1], ↑ |
| **Exact-match accuracy** | Fraction of fields where `ŷ_i == y_i` after normalization. | Identifiers, codes, dates, currency. | [0,1], ↑ |
| **Character / Word Error Rate (CER/WER)** | Normalized edit distance between predicted and gold string (char- or word-level). | Free-text values where partial credit matters. | [0,∞), ↓ |
| **ANLS** (Average Normalized Levenshtein Similarity) | For each field, NLS = 1 − normalized-Levenshtein(ŷ,y) if that similarity ≥ τ (typically τ=0.5), else 0; average over fields. Tolerates minor OCR noise while punishing real errors. | DocVQA-style value matching; text values. | [0,1], ↑ |
| **TEDS / TEDS-Struct / TEDS-Text** | Tree-Edit-Distance-based Similarity between HTML tree of predicted vs gold table: TEDS = 1 − d / max(\|T_pred\|,\|T_gt\|). TEDS-Struct ignores cell text (structure only); TEDS-Text adds char-level cell-content comparison. | Tables. | [0,1], ↑ |
| **GriTS** (Top / Con / Loc) | Grid Table Similarity: align predicted and gold cell **matrices** via Factored-2D-Most-Similar-Substructures, then aggregate pairwise cell similarity into precision/recall/F-score. Variants score topology (Top), content (Con), and location (Loc). | Tables, when matrix-native scoring is preferred over tree edits. | [0,1], ↑ |
| **JSON field scoring (schema-as-spec)** | Per-leaf scoring where each field declares its metric (exact / numeric-tolerance / semantic-equivalence); arrays aligned before scoring; **distinguish omission (missing gold field) from hallucination (predicted field with no gold)**. Report both an omission rate and a hallucination rate. | Nested JSON extraction (the VerifyDoc setting). | per-field, ↑ |

Implementation note: adopt ExtractBench's executable-schema pattern — annotate each schema leaf with its scoring rule so the evaluator is data-driven, not hard-coded. Report **hallucination rate** (spurious fields) and **omission rate** (dropped fields) separately; these are what the confidence/abstention layer must catch.

### 5.B Calibration metrics (does the confidence mean anything?)

Bin the `N` fields by confidence into `M` bins `B_m`.

- **Expected Calibration Error (ECE)** = Σ_m (|B_m|/N) · |acc(B_m) − conf(B_m)|. Use M=15 equal-width bins as the default; **also report Adaptive ECE** (equal-mass bins) because equal-width bins are unreliable when confidences cluster. Lower is better; ECE=0 is perfect calibration.
- **Maximum Calibration Error (MCE)** = max_m |acc(B_m) − conf(B_m)|. Worst-bin gap; report alongside ECE.
- **Brier score** = (1/N) Σ_i (c_i − correct_i)². Proper scoring rule; decomposes as Uncertainty − Resolution + Reliability (report the reliability component as a calibration cross-check). Lower is better.
- **Negative Log-Likelihood (NLL)** = −(1/N) Σ_i [correct_i·log c_i + (1−correct_i)·log(1−c_i)]. Penalizes confident errors hard.
- **Reliability diagram** — accuracy vs confidence per bin with the y=x reference; the primary calibration figure in the paper.
- **Target Calibration Error (TCE)** (recommended for the abstention framing) = E over target risks α of |R_test(α) − α|, where R_test(α) is the achieved selective risk when the acceptance threshold is tuned to hit target α. TCE measures whether you can actually *hit a requested error rate at an operating point*, which matters more to VerifyDoc than global [0,1] calibration.

Pitfalls to state in the paper: ECE is sensitive to binning scheme and bin count; it can look good while the model is useless at ranking errors — which is why calibration metrics (B) must always be reported **together with** selective-prediction metrics (C).

### 5.C Selective-prediction / abstention metrics (is "review vs accept" good?)

Sort fields by confidence descending. Let **coverage** = fraction accepted (answered) and **selective risk** = error rate among accepted.

- **Risk–Coverage (RC) curve** — plot selective risk vs coverage; the central selective-prediction figure.
- **AURC** (Area Under the RC curve) — integrates risk across all thresholds; lower is better; reflects how well confidence separates correct from incorrect fields.
- **E-AURC / Excess-AURC** = AURC − AURC_oracle, where the oracle ranks all correct fields above all incorrect ones. Removes the base-error-rate contribution so numbers are comparable across models/datasets (Geifman & El-Yaniv, 2019). Lower is better; report this as the primary abstention-quality scalar.
- **Coverage@Risk (C@R)** — the maximum coverage achievable while keeping selective risk ≤ target α (e.g., report Coverage@2% and Coverage@5%). **This is the headline product number** ("VerifyDoc auto-accepts X% of fields at a 2% error budget").
- **Risk@Coverage** — the selective risk at a fixed coverage (e.g., risk@80%).
- **Accuracy@k%** — accuracy on the top-k% most-confident fields (k ∈ {10,25,50}); an intuitive companion to C@R.
- **Error-detection framing (treat "field is wrong" as binary detection, score = 1−confidence):** **AUROC**, **AUPR**, and **FPR@95%TPR**. AUROC/AUPR summarize separability; FPR@95%TPR = fraction of correct fields wrongly flagged when catching 95% of errors — directly the "how much needless human review" number.
- **Human-review rate ↔ automation rate:** report the **straight-through-processing (STP) rate** = coverage at the target risk = fraction of fields needing no human touch. This is the metric a buyer cares about.

### 5.D Grounding / provenance metrics (is the value traceable to the page?)

- **Box grounding accuracy @ IoU τ** — fraction of fields whose predicted provenance box has IoU ≥ τ (report τ ∈ {0.5, 0.7}) with the gold source region.
- **Mean IoU** — average IoU over grounded fields (secondary).
- **Span grounding F1** — for text-layer provenance, token-span overlap (precision/recall/F1) between predicted and gold character spans.
- **Grounding-conditioned correctness** — accuracy of fields whose grounding is correct vs incorrect; used to test the hypothesis that *ungrounded values are more likely wrong* (i.e., grounding is itself a confidence signal), following the "visually grounded / geometrically verifiable" framing of risk-controlled generative OCR (arXiv:2603.19790).

### 5.E Current SOTA / reference numbers (to situate results; verify at write-up time)

| System / benchmark | Reported number | Note |
|---|---|---|
| Frontier LLMs on **ExtractBench** (complex schemas) | ~**4.6%** field-level pass rate; ~**0%** valid on 369-field schema | The reliability gap VerifyDoc targets (arXiv:2602.12247). |
| **PaddleOCR-VL-1.6** on OmniDocBench v1.6 | **96.33%** overall | ~0.9B params; vendor self-reported; runs on one consumer GPU. |
| **GLM-OCR** on OmniDocBench v1.5 | **94.6%** (SOTA; beats Gemini-3 Pro / GPT-5.2) | Parsing quality only — no calibration/abstention reported. |
| PP-OCRv6 hallucination benchmark (from earlier lit) | small specialist **93.2%** hallucination-free vs GPT-5.5 **78.0%**, Qwen3-VL-235B **80.6%** | Motivates confidence/abstention over raw accuracy. |
| Small OCR/VLM models for a single GPU | PaddleOCR-VL ~0.9B; DeepSeek-OCR ~570M active (MoE); Surya 2 ~650M (runs on RTX 4090 via GGUF); dots.ocr; GOT-OCR2.0 ~580M | The VerifyDoc default extractors. |
| Calibration/abstention numbers for document extraction | **Essentially none published** | This absence *is* the paper's opportunity — VerifyDoc reports the first. |

### 5.F Recommended metric suite (tell the complete story)

**Primary (headline table):**
1. Extraction quality: **Field-F1** (KIE sets) and **JSON per-field pass rate** with **hallucination/omission rates** (nested sets); **TEDS-Struct + GriTS-Con** for tables.
2. Calibration: **ECE (15-bin) + Adaptive ECE**, with the **reliability diagram**.
3. Abstention: **E-AURC** and **Coverage@2% / Coverage@5%** (STP rate).
4. Grounding: **Box grounding accuracy @ IoU 0.5**.

**Secondary (appendix / ablations):** MCE, Brier, NLL, TCE, full RC curve, AUROC/AUPR/FPR@95%TPR, Accuracy@{10,25,50}%, mean IoU, span-grounding F1, ANLS/CER for text values.

The narrative the suite must support: *"Raw extractor confidence is miscalibrated (high ECE) and ranks errors poorly (high E-AURC); VerifyDoc's consensus+grounding signal, with post-hoc calibration, cuts ECE by ≥X and raises Coverage@2% from Y% to Z%, so a human reviews only (100−Z)% of fields."*

### 5.G Expected baselines & ablations (what reviewers will demand)

Confidence-signal baselines (each fed through the same abstention policy for a fair RC comparison):
- **Raw token/sequence probability** (log-prob mean/min per field) — where logits are available.
- **Verbalized confidence** — ask the extractor to emit a 0–1 score per field.
- **Self-consistency / consensus** — sample the extractor *k* times (and/or run *m* different extractors) and use agreement as confidence (black-box; the likely VerifyDoc default).
- **Grounding-based** — confidence from provenance quality (does a supporting box/span exist; how tight).
- **VerifyDoc (combined)** — learned/ensembled combination of the above.

Calibration methods (applied on a held-out calibration split, compared head-to-head):
- **Temperature scaling**, **Platt scaling**, **isotonic regression**, **histogram binning**, and **split conformal / conformal risk control** for distribution-free risk guarantees (report both its guarantee and the abstention it forces, per arXiv:2606.29054).

Ablations: extractor choice (PaddleOCR-VL vs dots.ocr vs API VLM); *k* in self-consistency; with/without grounding signal; field-type breakdown (identifiers vs quantities vs free text); in-domain vs out-of-domain (train calibration on invoices, test on receipts) to test transfer.

### 5.H Experimental protocol

- **Datasets & splits (public, extend into VerifyDocBench):** FUNSD, CORD, SROIE, DocILE, XFUND (KIE); DocVQA (value matching); PubTabNet / PubTables-1M (tables); plus curated public PDFs (e.g., financial filings) for deep nested schemas. Use each dataset's official train/val/test split; reserve a dedicated **calibration split** distinct from test (calibration/abstention thresholds must never be tuned on test).
- **Licenses (verify before redistribution):** licenses vary (e.g., CORD is permissive; some sets are research/non-commercial). For the released VerifyDocBench, redistribute **your added per-field correctness labels + gold boxes + scripts** and reference the original downloads, rather than re-hosting restrictively licensed source images. State each source's license in the dataset card.
- **Ground-truth labeling:** where correctness/boxes aren't already present, label per-field correctness and source regions; use ≥2 annotators on a sample and report **inter-annotator agreement** (Cohen's/Fleiss' κ). Document the labeling guide in the repo.
- **Statistical rigor:** report **95% confidence intervals via bootstrap** (resample fields/documents) on every headline metric; for model-vs-model claims, report a paired bootstrap or permutation test. Fix and log all seeds; pin model versions/checkpoints.
- **Compute:** all default extractors run on a single consumer/cloud GPU (e.g., one 24 GB card); API models used only for comparison rows. The pipeline is inference + scoring — no training of large models required.
- **Reproducibility:** one command reproduces every table (`make results`); publish exact model versions, prompts, seeds, and dataset commit hashes.

---

## 6. System architecture

```
Document (PDF/img) + Schema
        │
        ▼
┌───────────────────────┐
│  Ingest / Renderer     │  pdf→images, text-layer extraction, page geometry
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Extractor Adapter     │  pluggable: PaddleOCR-VL | dots.ocr | Docling/MinerU
│  (model-agnostic)      │  output | API VLM.  k samples for self-consistency.
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Confidence Estimator  │  token-prob | verbalized | consensus | grounding-based
│                        │  → raw per-field score
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Calibrator            │  temperature/Platt/isotonic/histogram/conformal
│                        │  (fitted on calibration split)
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Grounder              │  attach {page,bbox} / {char_span}; compute IoU support
└──────────┬────────────┘
           ▼
┌───────────────────────┐
│  Abstention Policy     │  threshold for target risk α → accept | review
└──────────┬────────────┘
           ▼
   Verified JSON  ──►  Review UI (Streamlit): page render + green/yellow/red fields
        │
        ▼
   Eval Harness (verifydoc/eval): A/B/C/D metrics, RC curves, reliability diagrams
```

Design rules: adapters isolate all model-specific code behind one interface (`extract(doc, schema) -> list[FieldPrediction]`); the confidence, calibration, grounding, and abstention stages are independent and composable; the eval harness is decoupled and can score any system that emits the `FieldPrediction` schema (so it doubles as the benchmark scorer).

---

## 7. Repository structure

```
verifydoc/
├── README.md                     # quickstart + the killer demo GIF
├── CLAUDE.md                     #  working context (see separate file)
├── PROJECT.md                    # this document
├── LICENSE                       # Apache-2.0 (permissive → adoption)
├── pyproject.toml                # packaging (pip install verifydoc)
├── Makefile                      # make test | lint | results | demo
├── .pre-commit-config.yaml       # ruff + black + mypy hooks
├── .github/workflows/ci.yml      # lint + type + unit tests on PR
├── verifydoc/
│   ├── __init__.py
│   ├── types.py                  # FieldPrediction, Grounding, Document, Schema
│   ├── ingest/                   # pdf render, text-layer, geometry
│   ├── adapters/                 # extractor adapters (one file each)
│   │   ├── base.py               # ExtractorAdapter interface
│   │   ├── paddleocr_vl.py
│   │   ├── dots_ocr.py
│   │   ├── docling.py
│   │   └── api_vlm.py
│   ├── confidence/               # token_prob | verbalized | consensus | grounding
│   ├── calibration/              # temperature | platt | isotonic | histogram | conformal
│   ├── grounding/                # bbox/span attach + IoU utils
│   ├── policy/                   # abstention thresholds, target-risk solver
│   ├── pipeline.py               # wires the stages; the public verify() entrypoint
│   ├── cli.py                    # `verifydoc extract doc.pdf --schema s.json`
│   └── eval/                     # metrics + curves + significance
│       ├── extraction.py         # P/R/F1, exact, CER/WER, ANLS, TEDS, GriTS, JSON scoring
│       ├── calibration.py        # ECE, AdaptiveECE, MCE, Brier, NLL, TCE, reliability
│       ├── selective.py          # RC curve, AURC, E-AURC, C@R, Acc@k, AUROC/AUPR/FPR95
│       ├── grounding.py          # IoU accuracy, span F1
│       └── stats.py              # bootstrap CIs, paired tests
├── benchmark/                    # VerifyDocBench
│   ├── datasets/                 # loaders per source (FUNSD/CORD/SROIE/DocILE/XFUND/...)
│   ├── labeling/                 # annotation guide + tools + IAA scripts
│   ├── schemas/                  # per-document JSON schemas with per-field scoring rules
│   └── card.md                   # dataset card: sources, licenses, splits, stats
├── ui/streamlit_app.py           # review UI: render page, highlight fields, click-through
├── scripts/                      # reproduce tables/figures (make results calls these)
├── configs/                      # yaml configs for runs/ablations
├── tests/                        # unit tests mirroring verifydoc/ (pytest)
└── paper/                        # LaTeX; tables/figures auto-written by scripts
```

---

## 8. Tech stack & data sources

- **Language/runtime:** Python 3.11.
- **Core libs:** `pydantic` (typed `FieldPrediction`/schema), `pdfplumber`/`pypdfium2` (render + text layer), `pillow`/`opencv` (geometry), `numpy`/`scipy`/`scikit-learn` (calibration, isotonic, metrics), `pandas` (results), `matplotlib` (reliability + RC figures), `datasets` (loaders), `streamlit` (UI), `typer` (CLI).
- **Extractors:** PaddleOCR-VL and dots.ocr (local, single-GPU) as open defaults; Docling/MinerU output adapter; one API-VLM adapter for comparison rows. Grounded/bbox mode used where the model supports it.
- **Calibration/UQ:** implement temperature/Platt/isotonic/histogram in-repo; split conformal / conformal risk control for the guarantee row.
- **Public datasets:** FUNSD, CORD, SROIE, DocILE, XFUND, DocVQA, PubTabNet, PubTables-1M, plus curated public filings. (Verify licenses; redistribute annotations + scripts, not restricted source images.)

---

## 9. Implementation plan (6 weeks)

**Week 1 — skeleton + types + one adapter + eval stubs.** Repo, packaging, CI, pre-commit. Define `types.py` (`FieldPrediction`, `Grounding`). Implement ingest (PDF→images + text layer). Ship the **PaddleOCR-VL adapter** end-to-end on a handful of CORD receipts. Stub `eval/extraction.py` with Field-F1 + exact-match. *DoD: `verifydoc extract sample.pdf --schema cord.json` returns JSON with placeholder confidences; CI green.*

**Week 2 — confidence signals + full extraction metrics.** Implement token-prob, verbalized, and consensus (k-sample) confidence; add second adapter (dots.ocr). Complete `eval/extraction.py` (CER/WER, ANLS, TEDS/TEDS-Struct, GriTS, JSON per-field scoring with omission vs hallucination). *DoD: raw RC curve can be plotted for consensus confidence on CORD.*

**Week 3 — calibration + selective-prediction harness.** Implement `eval/calibration.py` (ECE, Adaptive ECE, MCE, Brier, NLL, TCE, reliability diagram) and `eval/selective.py` (RC, AURC, E-AURC, C@R, Acc@k, AUROC/AUPR/FPR95). Implement calibrators (temperature/Platt/isotonic/histogram). *DoD: first real table — raw vs temperature-scaled ECE and Coverage@2% on CORD, with bootstrap CIs.*

**Week 4 — grounding + abstention policy + conformal.** Grounder (attach bbox/span; IoU support), `eval/grounding.py`, target-risk abstention solver, split-conformal/CRC row. Test grounding-conditioned correctness hypothesis. *DoD: full A/B/C/D metrics on CORD + one KIE set.*

**Week 5 — VerifyDocBench + scale-out + UI.** Build dataset loaders + schemas + labeling for the benchmark; run all extractors × signals × calibrators across datasets via `configs/`; write the dataset card with licenses and IAA. Ship the **Streamlit review UI** (page render + green/yellow/red + click-through) — this is the demo. *DoD: `make results` reproduces every table/figure; UI records a shareable GIF.*

**Week 6 — paper + release.** Write the paper (§11) with auto-generated tables/figures; polish README with the GIF and 30-second quickstart; tag `v0.1.0`; publish to PyPI; post arXiv preprint; submit to a workshop (§11). *DoD: repo public + PyPI live + arXiv submitted + Show HN / Reddit / X launch posts drafted.*

---

## 10. Git workflow & engineering conventions

- **Repo from commit #1.** `git init`, Apache-2.0 license, `.gitignore` (Python + data + model caches). Never commit datasets/model weights — commit loaders and hashes.
- **Branching:** trunk-based. `main` always green. Short-lived feature branches `feat/<module>`, `fix/<thing>`, `docs/<thing>`, `exp/<ablation>`. Open a PR into `main`; squash-merge.
- **Conventional Commits:** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `bench:`. Example: `feat(calibration): add isotonic regression + ECE regression test`.
- **Every PR:** passes `ruff` (lint), `black` (format), `mypy` (types), and `pytest` (unit) via `.github/workflows/ci.yml`; adds/updates tests; updates docs if the public API changed. No PR merges red.
- **Testing:** unit tests mirror `verifydoc/`; **no network in unit tests** (mock adapters); metric functions get numeric regression tests against hand-computed fixtures (e.g., a 4-field toy set with known ECE/AURC). Target ≥80% coverage on `eval/`.
- **Reproducibility commits:** results are regenerated by `make results` from pinned configs/seeds; check in the generated tables under `paper/` so diffs are reviewable.
- **Releases:** semantic version tags (`v0.1.0`); `CHANGELOG.md` from commit history; PyPI publish on tag via a release workflow.
- **Issues/roadmap:** track the Week 1–6 DoDs as issues/milestones; label `good-first-issue` after launch to invite contributors (helps stars).

---

## 11. Paper plan

**Title (working):** *VerifyDoc: Calibrated, Abstaining, Grounded Document Extraction — A Benchmark and Strong Open Baseline.*

**Structure:** Introduction (the silent-hallucination gap; ExtractBench's own future-work call) → Related work (extraction benchmarks; calibration/selective prediction; conformal; grounded OCR) → Task formulation (§2) → VerifyDocBench (construction, labeling, IAA, licenses) → Method (confidence signals + calibration + grounding + abstention) → Experiments (metric suite §5.F; baselines/ablations §5.G) → Results (raw confidence is miscalibrated and ranks errors poorly; VerifyDoc improves ECE and Coverage@target) → Limitations (label scale; conformal's abstention cost; extractor coverage) → Broader impact → Release.

**Headline claims to earn:** (1) first calibration + selective-risk + grounding benchmark for document extraction; (2) consensus+grounding beats raw token-prob/verbalized confidence for error ranking; (3) with post-hoc calibration, VerifyDoc holds a target error rate while auto-accepting the large majority of fields (report the exact STP rate).

**Venues & realistic timing (submission window Aug–Oct 2026):** arXiv immediately on tag; then **NeurIPS 2026 workshops** (reliable/trustworthy-ML or document-AI; workshop-contribution dates in the late-Sept–Oct 2026 range), **EMNLP 2026 / AACL 2026 workshops**, and **TMLR** for a fast archival, citable full paper. Honest expectation: a **journal publication is not achievable in 3 months**; arXiv + a workshop acceptance + a well-starred repo is, and that combination is the usable evidence. (ExtractBench itself landed at SIGKDD 2026, confirming venue appetite for this exact topic.)

---

## 12. Risks & pivots

- **Extractor can't beat incumbents on raw parsing.** Irrelevant — VerifyDoc layers on top; its value (calibration + abstention + grounding) is orthogonal to parsing accuracy.
- **A parser ships native field-level confidence mid-project.** Pivot VerifyDoc to be *cross-tool*: the benchmark + a head-to-head calibration/abstention comparison across all of them (still novel and useful).
- **Labeling ground-truth correctness is slow.** Bootstrap from datasets that already carry gold field values (CORD/SROIE) so `correct_i` is computable automatically; hand-label only source boxes and hard nested schemas; report on what's labeled and expand post-launch.
- **The combined method doesn't beat simple consensus.** Then *consensus + conformal* is the honest, strong baseline and the **benchmark itself is the primary contribution** — arguably more citable.
- **Logit access missing for API models.** Lead with black-box signals (verbalized + consensus + grounding); logit-based rows become an ablation where available.

---

## References (select; verify at write-up)

- ExtractBench — arXiv:2602.12247 (SIGKDD 2026); code github.com/ContextualAI/extract-bench.
- LLMStructBench — arXiv:2602.14743. Structured Output Benchmark — arXiv:2604.25359.
- GriTS — arXiv:2203.12555 (ICDAR 2023). TEDS/PubTabNet — Zhong et al. OmniDocBench — CVPR 2025 (opendatalab/OmniDocBench).
- Risk-Controlled Generative OCR — arXiv:2603.19790.
- Conformal risk control for structured generation — arXiv:2606.29054; conformal factuality — Mohri & Hashimoto 2024; conformal abstention — Abbasi-Yadkori et al. 2024.
- Selective prediction / E-AURC — Geifman & El-Yaniv 2017/2019. Calibration/ECE — Guo et al. 2017; Kadavath et al. 2022.
- Model cards / leaderboards: PaddleOCR-VL, dots.ocr, DeepSeek-OCR, Surya 2, GOT-OCR2.0 (2025–2026).

*All external numbers (SOTA scores, star counts, model sizes, dataset licenses, deadlines) are point-in-time as of mid-2026 and must be re-verified before submission.*
