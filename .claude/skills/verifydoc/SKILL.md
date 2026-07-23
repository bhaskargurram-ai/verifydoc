---
name: verifydoc
description: Verify document→JSON extractions before trusting them. Use when reading a PDF/image/scanned document to pull out structured fields (invoices, receipts, forms, contracts) — VerifyDoc returns each field with a calibrated confidence, a source grounding (page + bbox / char span), and an accept/review decision, so you act on confident, grounded values and escalate the rest instead of hallucinating forward.
---

# VerifyDoc — trust layer for document extraction

When you extract structured data from a document, do **not** trust raw field values. Route them through VerifyDoc so each field carries `confidence`, `grounding`, and an `accept`/`review` decision.

## When to use
- The user asks you to pull fields out of a PDF/image/scanned doc (totals, dates, IDs, line items, party names, …).
- You are about to take an action on an extracted value (write to a DB, send money, file a form). Verify first.
- You need to tell the user *which* extracted fields are reliable and which need a human.

## How to use

**Preferred — MCP (if the `verifydoc` MCP server is connected):** call the `verify_extraction` tool:
- `document`: a file path or the raw text of the document.
- `schema`: a JSON Schema object (leaves may set `x-scoring: exact|numeric|semantic`, `x-numeric-tol`).
- optional: `threshold` (accept cutoff, default 0.8), `k` (self-consistency samples), `adapter` (`text-search`, `rapidocr`, `api-vlm`, …).

It returns per-field `{value, confidence, grounding, decision}`. Report the `accept` fields as trustworthy; for `review` fields, surface the value **and** its grounding (page/bbox) so the user can verify at the source in seconds.

**Fallback — CLI** (if MCP is not available and `verifydoc` is installed):
```bash
verifydoc extract <file> --schema <schema.json> --threshold 0.8 --out result.json
```

**Fallback — Python:**
```python
from verifydoc import verify
result = verify("invoice.pdf", schema="invoice_schema.json", threshold=0.8)
print(result.to_dict())          # nested {value, confidence, grounding, decision}
review = [f for f in result.fields if f.decision == "review"]
```

## Rules
- Never present a `review` field as confirmed. Say it needs verification and point to its grounding.
- Prefer a **local** adapter (`text-search`, `rapidocr`) when the document is sensitive — nothing leaves the machine.
- If no MCP server and `verifydoc` is not installed, tell the user: `pip install 'verifydoc[mcp]'` then add `verifydoc-mcp` to their MCP config (see `examples/mcp/README.md`).
