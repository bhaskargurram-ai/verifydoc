"""API-VLM adapter (comparison rows only; local models are the defaults).

Prompts a vision-language model API for schema-conforming JSON where every
leaf carries a 0-1 self-assessed confidence — the *verbalized* signal. The
client is injectable, so unit tests run with a fake and no network
(golden rule #5); without an injected client the Anthropic SDK is used.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Schema, flatten_json

_SYSTEM = (
    "You extract structured data from documents. Reply with ONLY a JSON object "
    "matching the requested schema, where every leaf value is replaced by "
    '{"value": ..., "confidence": <float 0-1 honestly reflecting P(correct)>}.'
)


class CompletionClient(Protocol):
    """Minimal client surface so tests can inject a fake."""

    def complete(self, system: str, prompt: str) -> str: ...


class APIVLMAdapter(ExtractorAdapter):
    name = "api-vlm"

    def __init__(
        self,
        client: CompletionClient | None = None,
        model: str = "claude-sonnet-5",
    ) -> None:
        self._client = client
        self._model = model

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        text = "\n\f\n".join(page.text or "" for page in doc.pages)
        prompt = (
            f"Schema (JSON Schema):\n{json.dumps(schema.raw)}\n\n"
            f"Document text:\n{text}\n\n"
            "Extract every schema field you can find."
        )
        raw = self._complete(_SYSTEM, prompt)
        return self._parse(raw)

    def _complete(self, system: str, prompt: str) -> str:
        if self._client is not None:
            return self._client.complete(system, prompt)
        return self._anthropic_complete(system, prompt)  # pragma: no cover - network

    def _anthropic_complete(self, system: str, prompt: str) -> str:  # pragma: no cover
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "APIVLMAdapter without an injected client requires: pip install anthropic"
            ) from exc
        response = anthropic.Anthropic().messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.content[0].text)

    @staticmethod
    def _parse(raw: str) -> list[FieldPrediction]:
        """Parse {leaf: {value, confidence}} JSON into predictions."""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            obj: Any = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return []
        flat = flatten_json(obj)
        by_field: dict[str, dict[str, Any]] = {}
        for path, value in flat.items():
            if path.endswith(".value"):
                by_field.setdefault(path[: -len(".value")], {})["value"] = value
            elif path.endswith(".confidence"):
                by_field.setdefault(path[: -len(".confidence")], {})["confidence"] = value
        preds = []
        for path, parts in by_field.items():
            if "value" not in parts:
                continue
            verbalized = parts.get("confidence")
            meta = {"verbalized_confidence": float(verbalized)} if verbalized is not None else {}
            preds.append(
                FieldPrediction(path=path, value=parts["value"], confidence=0.5, meta=meta)
            )
        return preds
