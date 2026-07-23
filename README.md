# VerifyDoc

> **The trust layer for document → structured-JSON extraction.** Wrap any extractor — get back JSON where **every field** carries a **calibrated confidence**, a **source grounding** (page + bbox / char span), and an **accept/review decision** tuned to your error budget.

[![CI](https://github.com/bhaskargurram-ai/verifydoc/actions/workflows/ci.yml/badge.svg)](https://github.com/bhaskargurram-ai/verifydoc/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/verifydoc.svg)](https://pypi.org/project/verifydoc/)
[![Downloads](https://img.shields.io/pypi/dm/verifydoc.svg)](https://pypi.org/project/verifydoc/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/bhaskargurram-ai/verifydoc?style=flat)](https://github.com/bhaskargurram-ai/verifydoc/stargazers)
[![Docs](https://img.shields.io/badge/docs-online-green.svg)](https://bhaskargurram-ai.github.io/verifydoc/)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://verifydoc-demo.web.app)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow.svg)](https://huggingface.co/spaces/bhaskargurram-ai/verifydoc)

**Every other parser tells you *what* it read; VerifyDoc tells you *which values to trust*.**

**▶ [Try the live demo](https://verifydoc-demo.web.app)** or **[🤗 the Hugging Face Space](https://huggingface.co/spaces/bhaskargurram-ai/verifydoc)** — no install: paste a receipt or upload a PDF and watch fields get accepted or routed to review. (Local extraction; for the Claude model, paste your own API key — it's used only for that request and never stored.)

> **🔒 Private by default — your documents never leave your machine.** Every extractor can run fully **local and offline** (RapidOCR, PaddleOCR, dots.ocr, Docling, or a local HF VLM); hosted API models are opt-in and comparison-only. Self-host the whole review app + API on your own infra, or call it from a **web app** or a **WhatsApp / Telegram bot** — the operator controls the data end to end.

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
pip install verifydoc                 # core (text pipelines + eval harness)
pip install 'verifydoc[pdf]'          # + PDF/image ingestion
uvx verifydoc extract doc.pdf --schema schema.json   # zero-install run (uv)
pipx install verifydoc                # isolated CLI install
docker run -p 8000:8000 ghcr.io/bhaskargurram-ai/verifydoc   # self-hosted API + web UI
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
verifydoc batch ./invoices --schema schema.json -o out/   # whole folder → one JSON/doc + summary.json
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

## Integrations

VerifyDoc is a **drop-in trust layer** — it wraps whatever you already use. No framework is a hard dependency; the framework integrations work by duck typing (any `pydantic.BaseModel`, any `str -> dict` callable).

| Where you work | How to add VerifyDoc | Guide |
|---|---|---|
| **Claude Code** | MCP server (`verifydoc-mcp`) + the bundled skill in `.claude/skills/verifydoc/` | [skill](.claude/skills/verifydoc/SKILL.md) · [MCP](examples/mcp/README.md) |
| **Codex / Cursor / Cline / Claude Desktop** | point the client at `verifydoc-mcp` (stdio MCP) | [copy-paste configs](examples/mcp/README.md) |
| **Instructor / Outlines / Marvin / Pydantic-AI** | `verify_instructor_result(text, obj)` — verify any extracted `BaseModel` | [quickstart](docs/QUICKSTART_INTEGRATIONS.md) |
| **LangChain** | `VerifiedExtractor(chain.invoke, schema=...)` | [quickstart](docs/QUICKSTART_INTEGRATIONS.md) |
| **LlamaIndex / DSPy / Haystack** | wrap your `str -> dict/BaseModel` step | [examples/](examples/) |
| **Any REST client / web / mobile** | self-hosted FastAPI server + web app + WhatsApp/Telegram bots | [deploy](docs/DEPLOY.md) |

See [`docs/QUICKSTART_INTEGRATIONS.md`](docs/QUICKSTART_INTEGRATIONS.md) for copy-paste snippets and [`examples/`](examples/) for runnable end-to-end scripts.

## How VerifyDoc compares

VerifyDoc doesn't replace your parser — it adds the trust layer none of them ship:

| | per-field confidence | calibrated | source grounding | accept/review abstention | open-source |
|---|:---:|:---:|:---:|:---:|:---:|
| **VerifyDoc** (on top of any extractor) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Docling / MinerU / Marker | ❌ | ❌ | boxes only\* | ❌ | ✅ |
| PaddleOCR / dots.ocr | raw score | ❌ | boxes | ❌ | ✅ |
| Box / Azure / Textract | ✅ | ❓ | ✅ | ❌ | ❌ |

<sub>\*layout parsers emit boxes but no per-field *correctness* signal. Full audit: [docs/USP.md](docs/USP.md).</sub>

## What's inside

| Layer | Modules | Status |
|---|---|---|
| **Adapters** (all model code isolated here) | mock · text-search · RapidOCR · PaddleOCR · dots.ocr · Docling/MinerU output · API-VLM (OpenAI/Anthropic) | ✅ |
| **Confidence signals** | token-prob · verbalized · consensus (k-sample voting, **adaptive-k budget control**) · grounding-based · **entailment (pluggable NLI)** · learned combiner | ✅ |
| **Calibrators** (fit on a dedicated split, never test) | temperature · Platt · isotonic · histogram · **split conformal** · **grounding-conditioned (Mondrian) conformal** (novel — recovers coverage a pooled threshold forfeits) | ✅ |
| **Grounding** | value → page/bbox/char-span attachment with support scores | ✅ |
| **Policy** | empirical & conformal accept thresholds for a target selective risk | ✅ |
| **Eval harness / VerifyDocBench scorer** | Field-F1 · exact · CER/WER · ANLS · TEDS/TEDS-Struct · GriTS · omission vs hallucination · ECE/Adaptive-ECE/MCE/Brier/NLL/TCE · RC/AURC/E-AURC/Coverage@Risk/AUROC/AUPR/FPR@95 · box IoU/span-F1/grounding-conditioned correctness · bootstrap CIs + paired tests | ✅ |

Every metric implements the exact definition in [PROJECT.md §5](PROJECT.md) with a hand-computed numeric regression test (382 tests, `eval/` coverage 97%).

## Results on real documents

Two independent real OCR extractors (**RapidOCR** and **PaddleOCR**) on real
**CORD** receipts and **FUNSD** forms, scored by the harness (regenerate the
full tables locally with `make results`):

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
make results     # regenerates every benchmark table/figure from configs/ (local output)
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
reliability diagrams `make results` produces for what "calibrated" actually
buys you.

## Roadmap

- [x] v0.1 — library + CLI + harness + synthetic benchmark slice + UI
- [x] v0.2 — CORD + FUNSD real slices with gold boxes; learned combiner; 1000× faster grounder
- [x] v0.3 — real-model results (RapidOCR + PaddleOCR on CORD/FUNSD)
- [x] v0.4 — vendor-neutral API-VLM extractor (OpenAI/Anthropic) with k-sample consensus; compilable paper with auto-generated tables
- [x] v0.5 — **novel method** (grounding-conditioned conformal, +0.50 coverage at fixed risk) + **MCP server** (agent trust layer) + real frontier-VLM results
- [x] v0.6 — method validated on **real data at scale** (FUNSD 24%→71% coverage at 2% risk); inter-annotator-agreement tooling (`verifydoc iaa`); numeric-aware grounding
- [x] v0.10 — **live hosted demo + 🤗 Hugging Face Space**; `verifydoc batch <dir>`; adaptive-k consensus; entailment-based grounding (NLI); array-leaf alignment; LangGraph review agent
- [ ] dots.ocr via vllm; SROIE / DocILE / XFUND slices; human-labeled correctness at scale
- [ ] Paper submission ([contributions welcome](CONTRIBUTING.md))

## Documentation

- [Framework integrations](docs/INTEGRATIONS.md) — drop-in trust layer for Instructor, Pydantic-AI, Outlines, LangChain
- [How it works](docs/how-it-works.md) — the pipeline, the abstention idea, why grounding is a trust signal
- [Related work & positioning](docs/RELATED_WORK.md) — how VerifyDoc compares to Beyond Logprobs, Cleanlab TLM, conformal factuality, and the commercial IDP stack
- [MCP for agents](docs/MCP.md) — client config + the registry publish runbook
- [GPU runbook](docs/REAL_MODELS.md) — reproduce the real-extractor rows
- [USP audit](docs/USP.md) · [Full spec](PROJECT.md)

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
