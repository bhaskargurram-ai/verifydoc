"""Core verify logic for the VerifyDoc Hugging Face Space.

Kept free of any Gradio import so it can be unit-tested offline; ``app.py``
builds the UI on top of these functions. Reuses ``verifydoc.verify`` — the Space
adds no extraction logic of its own.
"""

from __future__ import annotations

import json
from typing import Any

from verifydoc import verify
from verifydoc.adapters import get_adapter
from verifydoc.ingest import document_from_text

TABLE_HEADERS = ["field", "value", "confidence", "decision", "grounding"]

DEFAULT_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "vendor": {"type": "string", "x-scoring": "semantic"},
            "date": {"type": "string"},
            "total": {"type": "number", "x-numeric-tol": 0.01},
        },
    },
    indent=2,
)

_INVOICE = (
    "ACME Supplies Ltd\n"
    "Invoice #: INV-2024-0912\n"
    "Date: 2024-03-04\n"
    "Subtotal: 1,100.00\n"
    "Tax: 134.50\n"
    "Total: $1,234.50"
)
_RECEIPT = (
    "Corner Cafe\n" "Order 88 — 2024-05-01\n" "Flat white 4.20\n" "Muffin 3.50\n" "Total: 7.70"
)

# (document_text, schema_json) pairs for gr.Examples
EXAMPLES: list[list[str]] = [
    [_INVOICE, DEFAULT_SCHEMA],
    [
        _RECEIPT,
        json.dumps(
            {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string", "x-scoring": "semantic"},
                    "date": {"type": "string"},
                    "total": {"type": "number", "x-numeric-tol": 0.01},
                },
            },
            indent=2,
        ),
    ],
]


def resolve_adapter(name: str, api_key: str | None = None) -> Any:
    """Local adapters run offline; ``api-vlm`` needs a per-request Anthropic key.

    The key is used only for this call and never stored — mirroring the hosted
    demo's bring-your-own-key policy.
    """
    name = (name or "text-search").strip()
    if name == "api-vlm":
        if not api_key:
            raise ValueError(
                "The Claude (api-vlm) model needs your own Anthropic API key — "
                "paste it in the field on the left. It is used only for this request "
                "and never stored."
            )
        from verifydoc.adapters.api_vlm import AnthropicClient, APIVLMAdapter

        return APIVLMAdapter(client=AnthropicClient(api_key=api_key))
    return get_adapter(name)


def _highlights(text: str, fields: list[Any]) -> list[tuple[str, str | None]]:
    """Split the source into (segment, label) runs for gr.HighlightedText,
    labelling each grounded span with its accept/review decision."""
    spans = sorted(
        (f for f in fields if f.grounding and f.grounding.char_span),
        key=lambda f: f.grounding.char_span[0],
    )
    out: list[tuple[str, str | None]] = []
    cur = 0
    for f in spans:
        start, end = f.grounding.char_span
        if start < cur or end <= start:
            continue
        if start > cur:
            out.append((text[cur:start], None))
        out.append((text[start:end], f.decision))
        cur = end
    if cur < len(text):
        out.append((text[cur:], None))
    return out or [(text, None)]


def run_verify(
    text: str,
    schema_json: str,
    adapter_name: str = "text-search",
    threshold: float = 0.8,
    api_key: str = "",
) -> tuple[list[list[Any]], list[tuple[str, str | None]], str]:
    """Verify a pasted document → (rows, highlighted-source, summary markdown)."""
    if not text or not text.strip():
        return [], [("Paste or pick a sample document, then click Verify.", None)], ""
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        return [], [(f"Invalid schema JSON: {exc}", "review")], "⚠️ Fix the schema JSON."
    try:
        adapter = resolve_adapter(adapter_name, api_key or None)
        result = verify(
            document_from_text("doc", [text]),
            schema,
            adapter=adapter,
            threshold=float(threshold),
        )
    except Exception as exc:  # surface adapter/key errors in the UI, don't crash
        return [], [(str(exc), "review")], "⚠️ " + str(exc)

    rows: list[list[Any]] = []
    for f in result.fields:
        where = (
            f"page {f.grounding.page} · support {f.grounding.support:.2f}" if f.grounding else "—"
        )
        rows.append(
            [
                f.path,
                "" if f.value is None else str(f.value),
                round(float(f.confidence), 3),
                "✅ accept" if f.decision == "accept" else "⚠️ review",
                where,
            ]
        )
    n = len(result.fields)
    acc = result.n_accepted
    stp = f"{100 * acc / n:.0f}%" if n else "—"
    summary = (
        f"**{n}** fields · **{acc}** auto-accepted · **{result.n_review}** to review · "
        f"straight-through **{stp}**"
        if n
        else "No fields extracted for this schema."
    )
    return rows, _highlights(text, result.fields), summary
