"""VerifyDoc MCP server — a trust layer for AI agents that read documents.

Exposes VerifyDoc over the Model Context Protocol so any MCP-capable agent
(Claude Desktop, IDEs, custom agents) can extract structured data from a
document and get, for every field, a **calibrated confidence**, a **source
grounding**, and an **accept/review decision**. Agents stop trusting raw
extractions blindly: they auto-accept the confident, grounded fields and route
the uncertain ones to a human — with provenance attached.

Tools exposed:
- ``verify_extraction(document, schema, threshold?, k?)`` — extract + verify a
  document (text or file path) against a JSON schema; returns per-field
  value/confidence/grounding/decision plus accept/review counts.
- ``list_adapters()`` — the available extractor backends.

Run:  python -m verifydoc.mcp_server         (stdio transport)
Install (Claude Desktop etc.): see docs/MCP.md.

The MCP SDK is an optional dependency: ``pip install 'verifydoc[mcp]'``.
Everything below the transport is the ordinary ``verify()`` pipeline, so the
server has no logic of its own to test beyond argument marshalling
(``_run_verify`` is unit-tested offline).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifydoc.adapters import _REGISTRY, get_adapter
from verifydoc.pipeline import verify


def _run_verify(
    document: str,
    schema: str | dict[str, Any],
    threshold: float = 0.8,
    k: int = 1,
    adapter: str = "text-search",
) -> dict[str, Any]:
    """Marshal MCP tool args into a ``verify()`` call and return a JSON-able dict.

    ``document`` is either a path to an existing file or raw document text.
    ``schema`` is a JSON Schema (dict or JSON string / path).
    """
    doc_arg: Any = document
    if not (len(document) < 1024 and Path(document).expanduser().exists()):
        # treat as raw text: write to an in-memory Document via the text loader
        from verifydoc.ingest import document_from_text

        doc_arg = document_from_text("mcp-input", [document])

    schema_arg: Any = schema
    if isinstance(schema, str):
        p = Path(schema)
        schema_arg = (
            json.loads(p.read_text(encoding="utf-8"))
            if (len(schema) < 1024 and p.exists())
            else json.loads(schema)
        )

    result = verify(doc_arg, schema_arg, adapter=get_adapter(adapter), k=k, threshold=threshold)
    return {
        "doc_id": result.doc_id,
        "n_accepted": result.n_accepted,
        "n_review": result.n_review,
        "threshold": result.threshold,
        "fields": [
            {
                "path": f.path,
                "value": f.value,
                "confidence": round(f.confidence, 4),
                "decision": f.decision,
                "grounding": f.grounding.model_dump() if f.grounding else None,
            }
            for f in result.fields
        ],
        "summary": (
            f"{result.n_accepted} field(s) auto-accepted, {result.n_review} routed to review. "
            "Review the flagged fields against their grounding before trusting them."
        ),
    }


def build_server() -> Any:  # pragma: no cover - requires the mcp extra
    """Construct the FastMCP server with VerifyDoc's tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("The MCP server needs: pip install 'verifydoc[mcp]'") from exc

    mcp = FastMCP("verifydoc")

    @mcp.tool()
    def verify_extraction(
        document: str,
        schema: str,
        threshold: float = 0.8,
        k: int = 1,
        adapter: str = "text-search",
    ) -> str:
        """Extract structured fields from a document and verify each one.

        Returns JSON where every field carries a calibrated confidence, a source
        grounding (page/bbox/char-span), and an accept/review decision. Use this
        before trusting any extracted value: accept the accepted, verify the rest.

        Args:
            document: a file path or the raw document text.
            schema: a JSON Schema (object) as a JSON string or file path; leaves
                may declare ``x-scoring`` (exact|numeric|semantic).
            threshold: confidence cutoff for auto-accept (default 0.8).
            k: self-consistency samples; k>1 enables consensus (default 1).
            adapter: extractor backend (default ``text-search``).
        """
        return json.dumps(_run_verify(document, schema, threshold, k, adapter), ensure_ascii=False)

    @mcp.tool()
    def list_adapters() -> str:
        """List the available extractor backends (adapters)."""
        return json.dumps(sorted(_REGISTRY))

    return mcp


def main() -> None:  # pragma: no cover - runs the server
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
