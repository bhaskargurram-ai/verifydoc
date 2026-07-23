"""Offline tests for the local HF-VLM adapter (fake generation client, no torch)."""

from verifydoc.adapters import _REGISTRY, get_adapter
from verifydoc.adapters.hf_vlm import HFVLMAdapter
from verifydoc.ingest import document_from_text
from verifydoc.types import Schema

SCHEMA = Schema.from_json_schema(
    {
        "type": "object",
        "properties": {
            "total": {"type": "number"},
            "vendor": {"type": "string", "x-scoring": "semantic"},
        },
    }
)


class _FakeGen:
    """Stand-in local model: returns a fixed JSON completion."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.seen: dict[str, str] = {}

    def generate(self, system: str, prompt: str) -> str:
        self.seen = {"system": system, "prompt": prompt}
        return self.raw


def test_extract_parses_fields_and_confidence():
    raw = '{"total": {"value": 7.7, "confidence": 0.92}, "vendor": {"value": "Cafe", "confidence": 0.8}}'
    fake = _FakeGen(raw)
    adapter = HFVLMAdapter(client=fake)
    doc = document_from_text("d", ["Cafe\nTotal: 7.70"])
    preds = {p.path: p for p in adapter.extract(doc, SCHEMA)}
    assert set(preds) == {"total", "vendor"}
    assert preds["total"].value == 7.7
    assert preds["total"].meta["verbalized_confidence"] == 0.92
    # the schema + document text were both put in the prompt
    assert "total" in fake.seen["prompt"] and "Cafe" in fake.seen["prompt"]


def test_bad_json_yields_no_predictions():
    adapter = HFVLMAdapter(client=_FakeGen("sorry, I can't help with that"))
    doc = document_from_text("d", ["x"])
    assert adapter.extract(doc, SCHEMA) == []


def test_registered_and_constructible():
    assert "hf-vlm" in _REGISTRY
    adapter = get_adapter("hf-vlm")  # constructs without touching torch
    assert adapter.name == "hf-vlm"
