# VerifyDoc

**The trust layer for document → structured-JSON extraction.** Wrap any
extractor and get back JSON where **every field** carries a **calibrated
confidence**, a **source grounding** (page + bbox / char-span), and an
**accept/review decision** tuned to your error budget.

> Every other parser tells you *what* it read; VerifyDoc tells you *which values to trust*.

## Install

```bash
pip install verifydoc            # core
pip install 'verifydoc[pdf]'     # + PDF/image ingestion
```

## 30-second example

```python
from verifydoc import verify

result = verify("invoice.pdf", schema="invoice_schema.json", threshold=0.8)
for f in result.fields:
    print(f.path, f.value, round(f.confidence, 2), f.decision)   # accept / review
```

## Where to go next

- **[How it works](how-it-works.md)** — the ingest → adapter → confidence →
  calibration → grounding → policy pipeline.
- **[Integrations](INTEGRATIONS.md)** — drop VerifyDoc into Instructor, LangChain,
  LlamaIndex, Pydantic-AI, or any MCP agent (Claude Code, Cursor, Codex).
- **[Self-host & bots](DEPLOY.md)** — the FastAPI server, web review app, and
  Telegram/WhatsApp bots; runs on your own infra.
- **[Why VerifyDoc](USP.md)** — how it compares to Docling/MinerU/Marker and the
  commercial APIs, and what "calibrated confidence" actually buys you.

Privacy-first: every extractor can run fully local and offline; hosted API
models are opt-in and comparison-only.
