---
title: VerifyDoc
emoji: 🔒
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Which extracted document values should you trust?
tags:
  - document-ai
  - ocr
  - information-extraction
  - calibration
  - hallucination-detection
---

# VerifyDoc — the trust layer for document → JSON extraction

Every document parser tells you **what** it read. **VerifyDoc tells you which
fields to trust**: each extracted value comes back with a **calibrated
confidence**, a **source grounding** (page + span), and an **accept / review**
decision tuned to an error budget.

Paste a receipt or invoice, pick a schema, and click **Verify** — confident,
grounded fields are auto-accepted; the rest are routed to review with their
source location highlighted.

- Local extraction (text-search / RapidOCR) runs in the Space for free.
- The **Claude** model is **bring-your-own-key**: paste your Anthropic API key;
  it is used only for that request and never stored.

## Links

- ⭐ **GitHub:** <https://github.com/bhaskargurram-ai/verifydoc>
- 📦 **PyPI:** `pip install verifydoc` — <https://pypi.org/project/verifydoc/>
- 📖 **Docs:** <https://bhaskargurram-ai.github.io/verifydoc/>

## Run this Space locally

```bash
pip install -r requirements.txt
python app.py
```

Private by default — self-host the whole review app + API on your own infra:
`docker run -p 8000:8000 ghcr.io/bhaskargurram-ai/verifydoc`.
