"""API-VLM adapter (comparison rows; local models are the defaults).

Prompts a hosted vision-language / language model for schema-conforming JSON
where every leaf carries a 0-1 self-assessed confidence — the *verbalized*
signal. Two things make this the useful comparison row for the paper:

- **Structured output:** unlike the OCR-pipeline adapters (which read a text
  layer and label-search), the model returns the schema fields directly, so
  extraction quality is competitive.
- **A fair k-sample comparison:** ``extract_samples(k)`` samples at
  ``temperature > 0``, so self-consistency *consensus* and *verbalized*
  confidence are both non-degenerate (they are trivially flat for a
  deterministic OCR pipeline at k=1).

The provider is pluggable and vendor-neutral: pass any object implementing
``CompletionClient``, or select a built-in ``provider`` (``openai`` |
``anthropic``). Tests inject a fake client, so no network is required
(golden rule #5).
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

_DEFAULT_MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5"}


class CompletionClient(Protocol):
    """Minimal provider surface: prompt in, text out. Tests inject a fake."""

    def complete(self, system: str, prompt: str, temperature: float = 0.0) -> str: ...


class OpenAIClient:
    """OpenAI chat-completions client (lazy import)."""

    def __init__(self, model: str = "gpt-4o") -> None:  # pragma: no cover - network
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAIClient requires: pip install openai") from exc
        self._client = OpenAI()
        self._model = model

    def complete(
        self, system: str, prompt: str, temperature: float = 0.0
    ) -> str:  # pragma: no cover
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicClient:
    """Anthropic messages client (lazy import)."""

    def __init__(self, model: str = "claude-sonnet-5") -> None:  # pragma: no cover - network
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("AnthropicClient requires: pip install anthropic") from exc
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(
        self, system: str, prompt: str, temperature: float = 0.0
    ) -> str:  # pragma: no cover
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def default_client(provider: str = "anthropic", model: str | None = None) -> CompletionClient:
    """Build a built-in client for a provider name."""
    provider = provider.lower()
    if provider not in _DEFAULT_MODELS:
        raise ValueError(f"unknown provider {provider!r}; choose from {sorted(_DEFAULT_MODELS)}")
    model = model or _DEFAULT_MODELS[provider]
    if provider == "openai":
        return OpenAIClient(model)  # pragma: no cover - network
    return AnthropicClient(model)  # pragma: no cover - network


class APIVLMAdapter(ExtractorAdapter):
    name = "api-vlm"

    def __init__(
        self,
        client: CompletionClient | None = None,
        provider: str = "anthropic",
        model: str | None = None,
        sample_temperature: float = 0.7,
    ) -> None:
        self._client = client
        self._provider = provider
        self._model = model
        self._sample_temperature = sample_temperature

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        return self._extract(doc, schema, temperature=0.0)

    def extract_samples(
        self, doc: Document, schema: Schema, k: int = 1
    ) -> list[list[FieldPrediction]]:
        """k independent samples; k>1 uses temperature>0 for real consensus."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if k == 1:
            return [self._extract(doc, schema, temperature=0.0)]
        return [self._extract(doc, schema, temperature=self._sample_temperature) for _ in range(k)]

    def _extract(self, doc: Document, schema: Schema, temperature: float) -> list[FieldPrediction]:
        text = "\n\f\n".join(page.text or "" for page in doc.pages)
        prompt = (
            f"Schema (JSON Schema):\n{json.dumps(schema.raw)}\n\n"
            f"Document text:\n{text}\n\n"
            "Extract every schema field you can find."
        )
        raw = self._complete(_SYSTEM, prompt, temperature)
        return self._parse(raw)

    def _complete(self, system: str, prompt: str, temperature: float) -> str:
        client = self._client or default_client(self._provider, self._model)
        return client.complete(system, prompt, temperature)

    @staticmethod
    def _parse(raw: str) -> list[FieldPrediction]:
        """Parse ``{leaf: {value, confidence}}`` JSON into predictions."""
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
