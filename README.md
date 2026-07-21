# VerifyDoc

> **A trust layer for document → structured-JSON extraction.** Wrap any extractor; get back JSON where every field carries a **calibrated confidence**, a **source grounding** (page + bbox / char span), and an **accept/review decision** tuned to a target error rate.

[![CI](https://github.com/bhaskargurram-ai/verifydoc/actions/workflows/ci.yml/badge.svg)](https://github.com/bhaskargurram-ai/verifydoc/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

Modern VLM parsers read documents at 96%+ benchmark accuracy — and still produce **fluent, plausible, silently-wrong values** ($42.50 → $45.20) with no per-field signal telling you which values to trust. VerifyDoc doesn't compete with the parsers; it **layers on top of any of them** and adds the missing reliability contract:

- **Calibrated per-field confidence** — a probability that actually means something (ECE-verified).
- **Provenance** — every value points back to the page region / text span it was read from.
- **Abstention** — an accept/review policy that holds selective risk below your error budget, so a human reviews the ~5% of fields that are actually wrong instead of eyeballing all of them.

## Quickstart

```bash
pip install verifydoc
```

```python
from verifydoc import verify

result = verify("invoice.pdf", schema="invoice_schema.json")
for field in result.fields:
    print(field.path, field.value, f"{field.confidence:.2f}", field.decision)
```

```bash
verifydoc extract invoice.pdf --schema invoice_schema.json --target-risk 0.02
```

## Status

Under active development — see [PROJECT.md](PROJECT.md) for the full spec and roadmap.
