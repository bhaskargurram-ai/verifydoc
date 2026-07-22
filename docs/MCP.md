# VerifyDoc as an MCP server — a trust layer for AI agents

In the agentic era, AI agents extract structured data from documents and act
on it. The danger is that a fluent model returns a **plausible, wrong** value
(`$42.50` → `$45.20`) and the agent proceeds as if it were true. VerifyDoc's
MCP server gives any MCP-capable agent a trust contract: every field comes
back with a **calibrated confidence**, a **source grounding**, and an
**accept/review decision** — so the agent auto-accepts what's confident and
grounded, and escalates the rest instead of hallucinating forward.

## Install & run

```bash
pip install 'verifydoc[mcp]'
verifydoc-mcp            # stdio MCP server
```

## Connect it to an agent (Claude Desktop example)

Add to your MCP client config (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "verifydoc": {
      "command": "verifydoc-mcp"
    }
  }
}
```

The agent now has two tools:

- **`verify_extraction(document, schema, threshold?, k?, adapter?)`** —
  `document` is a file path or raw text; `schema` is a JSON Schema (leaves may
  declare `x-scoring`: `exact` | `numeric` | `semantic`). Returns JSON with,
  per field: `value`, `confidence`, `grounding` (`page`/`bbox`/`char_span`),
  and `decision` (`accept`/`review`), plus accept/review counts.
- **`list_adapters()`** — the available extractor backends.

## What the agent gets back

```json
{
  "n_accepted": 5,
  "n_review": 1,
  "fields": [
    {"path": "total", "value": "$1,234.50", "confidence": 0.98,
     "decision": "accept",
     "grounding": {"page": 0, "bbox": [0.10, 0.24, 0.21, 0.28], "support": 1.0}},
    {"path": "tax_id", "value": "XX-999", "confidence": 0.41,
     "decision": "review", "grounding": null}
  ],
  "summary": "5 field(s) auto-accepted, 1 routed to review. Review the flagged fields against their grounding before trusting them."
}
```

The agent policy becomes simple and safe: **act on `accept` fields; surface
`review` fields (with their grounding) to a human or a verification step.**

## Why this matters for agents

- **No silent hallucinations enter downstream actions** — ungrounded / low-
  confidence values are flagged, not trusted.
- **Provenance for every value** — the agent (or a human) can trace a field to
  the exact page region before acting on it.
- **Model-agnostic** — swap the `adapter` (OCR pipeline, VLM API, your own)
  without changing the agent contract.
