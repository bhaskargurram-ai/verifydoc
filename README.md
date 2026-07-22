# VerifyDoc

> **The trust layer for document → structured-JSON extraction.** Wrap any extractor — get back JSON where **every field** carries a **calibrated confidence**, a **source grounding** (page + bbox / char span), and an **accept/review decision** tuned to your error budget.

[![CI](https://github.com/bhaskargurram-ai/verifydoc/actions/workflows/ci.yml/badge.svg)](https://github.com/bhaskargurram-ai/verifydoc/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![VerifyDoc demo: a silently-wrong total is caught by grounding and routed to review](docs/demo.gif)

*Above: a real pipeline run (`scripts/make_demo_gif.py`). The extractor returned `$1,432.50`; the page says `$1,234.50`. Grounding support drops to 0.78, the field misses the accept threshold, and the reviewer is pointed at the exact source region. The other three fields are auto-accepted.*

## The problem

Modern document parsers read pages at 96%+ benchmark accuracy — and still emit **fluent, plausible, silently-wrong values** (`$42.50` → `$45.20`) with **no reliable per-field signal telling you which values to trust**. Commercial APIs (Box, Azure, Textract) sell field-level confidence as a closed feature. No popular open-source parser leads with it. ([full USP audit](docs/USP.md))

VerifyDoc doesn't compete with the parsers — it **layers on top of any of them**:

```
document + schema ─► ingest ─► extractor adapter ─► confidence ─► calibration
                                (any model)          signals       (fit on cal split)
                     ─► grounding ─► abstention policy ─► verified JSON + review UI
                        (bbox/span)   (target risk α)
```

At a chosen operating point, VerifyDoc auto-accepts as many fields as possible while holding the error rate among accepted fields below your target (e.g. ≤ 2%) — everything else is routed to `review` **with its source location attached**, so a human verifies in seconds instead of eyeballing every field.

## Quickstart

```bash
pip install verifydoc          # core (text pipelines + eval harness)
pip install 'verifydoc[pdf]'   # + PDF/image ingestion
```

```python
from verifydoc import verify

# ready-to-run sample lives in examples/
result = verify("examples/invoice.txt", schema="examples/invoice_schema.json")
for f in result.fields:
    print(f"{f.path:12} = {f.value!r:24} conf={f.confidence:.2f} {f.decision}")
    if f.grounding:
        print(f"             └─ page {f.grounding.page}, bbox {f.grounding.bbox}")
```

```bash
verifydoc extract examples/invoice.txt --schema examples/invoice_schema.json --threshold 0.8
streamlit run ui/streamlit_app.py     # review UI: green/red fields + click-through to source
```

See [`examples/`](examples/) for the full runnable walk-through.

## For AI agents (MCP)

Give any MCP-capable agent (Claude Desktop, IDEs, custom agents) a **trust
layer for reading documents** — so it acts on confident, grounded fields and
escalates the rest instead of hallucinating forward:

```bash
pip install 'verifydoc[mcp]'
verifydoc-mcp     # stdio MCP server exposing verify_extraction()
```

Every field the agent extracts comes back with `confidence + grounding +
accept/review`. See [docs/MCP.md](docs/MCP.md) for the one-line client config.

Schemas are plain JSON Schema, with each leaf optionally declaring **how it is scored** (the executable-schema pattern):

```json
{
  "type": "object",
  "properties": {
    "invoice_id": {"type": "string"},
    "vendor":     {"type": "string", "x-scoring": "semantic"},
    "total":      {"type": "number", "x-numeric-tol": 0.01}
  }
}
```

## What's inside

| Layer | Modules | Status |
|---|---|---|
| **Adapters** (all model code isolated here) | mock · text-search · RapidOCR · PaddleOCR · dots.ocr · Docling/MinerU output · API-VLM (OpenAI/Anthropic) | ✅ |
| **Confidence signals** | token-prob · verbalized · consensus (k-sample voting) · grounding-based · combined | ✅ |
| **Calibrators** (fit on a dedicated split, never test) | temperature · Platt · isotonic · histogram · **split conformal** · **grounding-conditioned (Mondrian) conformal** (novel — recovers coverage a pooled threshold forfeits) | ✅ |
| **Grounding** | value → page/bbox/char-span attachment with support scores | ✅ |
| **Policy** | empirical & conformal accept thresholds for a target selective risk | ✅ |
| **Eval harness / VerifyDocBench scorer** | Field-F1 · exact · CER/WER · ANLS · TEDS/TEDS-Struct · GriTS · omission vs hallucination · ECE/Adaptive-ECE/MCE/Brier/NLL/TCE · RC/AURC/E-AURC/Coverage@Risk/AUROC/AUPR/FPR@95 · box IoU/span-F1/grounding-conditioned correctness · bootstrap CIs + paired tests | ✅ |

Every metric implements the exact definition in [PROJECT.md §5](PROJECT.md) with a hand-computed numeric regression test (201 tests, `eval/` coverage 98%).

## Results on real documents

Two independent real OCR extractors (**RapidOCR** and **PaddleOCR**) on real
**CORD** receipts and **FUNSD** forms, scored by the harness
([full tables + reading](paper/generated/REAL_MODELS_RESULTS.md)):

| Confidence signal | ranks errors? | CORD AUROC (RapidOCR / PaddleOCR) |
|---|---|---|
| **learned combiner** | ✅ best | **0.89 / 0.84** |
| **grounding** | ✅ strong | 0.82 / 0.74 |
| token-probability | ~ moderate | 0.69 / 0.68 |
| verbalized / consensus | ✗ uninformative | 0.50 / 0.50 |

- **Grounding is a real trust signal:** grounded fields are **84–85% correct
  vs ~1%** for ungrounded (gap ≈ +0.84; box accuracy @IoU 0.5 = 0.75–0.78).
- **The abstention layer is honest:** with a weak field-extractor the base
  error rate is high, so conformal abstention at a 2–5% budget correctly
  refuses to auto-accept — you report *selective risk*, not headline accuracy.
- The synthetic slice (strong extractor) shows the other end: Coverage@2% ≈ 1.0.

The thesis holds on real data: **grounding + a learned fusion rank errors;
self-reported and single-sample-consensus confidence do not.**

## The benchmark

```bash
make results     # regenerates every table/figure in paper/generated from configs/
```

The harness runs **signals × calibrators × the full metric suite** with a
document-level calibration split (disjointness asserted in code), bootstrap
CIs, and a conformal-guarantee row. It ships a deterministic **synthetic**
slice (runs in CI) plus **CORD** and **FUNSD** loaders with gold source boxes;
`extractor:` dispatches to any adapter (`rapidocr`, `paddleocr-vl`, …) and
`dataset:` to any slice. See the [GPU runbook](docs/REAL_MODELS.md) to
reproduce the real-model rows. Core claims (grounding beats verbalized;
conformal holds its guarantee) are also **CI-enforced as unit tests**, not
just stated.

## Why not just use the parser's own score?

Because it doesn't exist (Docling/MinerU/Marker), or it's a raw recognition
score that was never calibrated against field-level correctness
(PaddleOCR/dots.ocr). See [docs/USP.md](docs/USP.md) for the audit, and the
reliability diagrams in `paper/generated/` for what "calibrated" actually
buys you.

## Roadmap

- [x] v0.1 — library + CLI + harness + synthetic benchmark slice + UI
- [x] v0.2 — CORD + FUNSD real slices with gold boxes; learned combiner; 1000× faster grounder
- [x] v0.3 — real-model results (RapidOCR + PaddleOCR on CORD/FUNSD)
- [x] v0.4 — vendor-neutral API-VLM extractor (OpenAI/Anthropic) with k-sample consensus; compilable paper with auto-generated tables
- [x] v0.5 — **novel method** (grounding-conditioned conformal, +0.50 coverage at fixed risk) + **MCP server** (agent trust layer) + real frontier-VLM results
- [x] v0.6 — method validated on **real data at scale** (FUNSD 24%→71% coverage at 2% risk); inter-annotator-agreement tooling (`verifydoc iaa`); numeric-aware grounding
- [ ] dots.ocr via vllm; SROIE / DocILE / XFUND slices; human-labeled correctness at scale
- [ ] Paper submission ([contributions welcome](CONTRIBUTING.md))

## Documentation

- [Framework integrations](docs/INTEGRATIONS.md) — drop-in trust layer for Instructor, Pydantic-AI, Outlines, LangChain
- [How it works](docs/how-it-works.md) — the pipeline, the abstention idea, why grounding is a trust signal
- [Related work & positioning](docs/RELATED_WORK.md) — how VerifyDoc compares to Beyond Logprobs, Cleanlab TLM, conformal factuality, and the commercial IDP stack
- [Real-model results](paper/generated/REAL_MODELS_RESULTS.md) — RapidOCR + PaddleOCR numbers and reading
- [GPU runbook](docs/REAL_MODELS.md) — reproduce the real-extractor rows
- [USP audit](docs/USP.md) · [Paper draft](paper/main.tex) · [Full spec](PROJECT.md)

## Development

```bash
git clone https://github.com/bhaskargurram-ai/verifydoc && cd verifydoc
uv venv .venv && uv pip install -e ".[dev]"
make test lint typecheck     # all green before any PR (CI enforces)
make results                 # regenerate benchmark tables + LaTeX
make paper                   # compile the paper (needs a LaTeX toolchain)
```

Contributions welcome — see the issues tagged `good-first-issue`. All model-specific code goes in `verifydoc/adapters/`; a new extractor is one file.

## Citation

```bibtex
@software{verifydoc2026,
  author = {Gurram, Bhaskar},
  title  = {VerifyDoc: Calibrated, Abstaining, Grounded Document Extraction},
  year   = {2026},
  url    = {https://github.com/bhaskargurram-ai/verifydoc}
}
```

Apache-2.0.
