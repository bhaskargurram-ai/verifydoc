"""Local Hugging Face model adapter — private extraction, no API.

Runs a **local** generative model (default ``Qwen/Qwen2.5-3B-Instruct``) over the
document's text layer and returns schema fields with a self-assessed confidence
— the same structured-output contract as the API-VLM adapter, but fully on-box:
nothing leaves the machine. Needs a local-model install:

    pip install 'verifydoc[hf]'      # transformers + torch

The generation backend is pluggable: pass any object implementing
``GenerationClient`` (``generate(system, prompt) -> str``). Tests inject a fake,
so no model / torch is required to exercise the adapter (golden rule #5).

# DECISION: this adapter reads the text layer (like api-vlm), so it works for any
# local instruct model; passing page images to a true vision model (Qwen2-VL) is
# a follow-up that only changes the client, not the adapter contract.
"""

from __future__ import annotations

import json
from typing import Protocol

from verifydoc.adapters.api_vlm import _SYSTEM, APIVLMAdapter
from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Schema

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class GenerationClient(Protocol):
    """Minimal local-generation surface: prompt in, text out. Tests inject a fake."""

    def generate(self, system: str, prompt: str) -> str: ...


class HFTransformersClient:  # pragma: no cover - heavy local model
    """Local text-generation via transformers (lazy import)."""

    def __init__(self, model: str = DEFAULT_MODEL, max_new_tokens: int = 1024) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("HFTransformersClient requires: pip install 'verifydoc[hf]'") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map="auto"
        )
        self._torch = torch
        self._max_new_tokens = max_new_tokens

    def generate(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True))


class HFVLMAdapter(ExtractorAdapter):
    """Extractor backed by a local HF generative model (private; no network)."""

    name = "hf-vlm"

    def __init__(self, client: GenerationClient | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        text = "\n\f\n".join(page.text or "" for page in doc.pages)
        prompt = (
            f"Schema (JSON Schema):\n{json.dumps(schema.json_schema)}\n\n"
            f"Document text:\n{text}\n\n"
            "Extract every schema field you can find."
        )
        client = self._client or HFTransformersClient(self._model or DEFAULT_MODEL)
        raw = client.generate(_SYSTEM, prompt)
        return APIVLMAdapter._parse(raw)
