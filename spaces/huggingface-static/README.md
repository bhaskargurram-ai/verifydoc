---
title: VerifyDoc
emoji: 🔒
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
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

A **free static Space**: the VerifyDoc review cockpit, calling the hosted
VerifyDoc API. Paste a receipt or invoice, pick a schema, click **Verify** —
every field comes back with a **calibrated confidence**, a **source grounding**,
and an **accept / review** decision. Confident, grounded fields are
auto-accepted; the rest are highlighted for review.

- Local extraction runs on the hosted backend for free.
- The **Claude** model is **bring-your-own-key**: paste your Anthropic key; it is
  used only for that request and never stored.

`index.html` is generated from the packaged review cockpit
(`verifydoc/server/review.html`) by `scripts/build_hf_static.py`, with its API
calls pointed at the hosted backend.

## Links

- ⭐ **GitHub:** <https://github.com/bhaskargurram-ai/verifydoc>
- 📦 **PyPI:** `pip install verifydoc` — <https://pypi.org/project/verifydoc/>
- 📖 **Docs:** <https://bhaskargurram-ai.github.io/verifydoc/>

Private by default — self-host the whole app on your own infra:
`docker run -p 8000:8000 ghcr.io/bhaskargurram-ai/verifydoc`.
