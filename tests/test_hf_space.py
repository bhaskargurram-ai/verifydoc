"""Offline smoke tests for the Hugging Face Space core (no gradio, no network).

Loads spaces/huggingface/core.py by path and drives run_verify with the
text-search adapter so the Space's logic is CI-verified even though gradio and
the hosted runtime are not installed here.
"""

import importlib.util
from pathlib import Path

import pytest

_CORE = Path(__file__).resolve().parents[1] / "spaces" / "huggingface" / "core.py"
_spec = importlib.util.spec_from_file_location("hf_space_core", _CORE)
assert _spec and _spec.loader
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


class TestRunVerify:
    def test_invoice_example_returns_trust_rows(self):
        text, schema = core.EXAMPLES[0]
        rows, highlights, summary = core.run_verify(text, schema, "text-search", 0.8)
        assert rows, "expected at least one field row"
        # each row is [field, value, confidence, decision, grounding]
        assert all(len(r) == len(core.TABLE_HEADERS) for r in rows)
        decisions = {r[3] for r in rows}
        assert decisions <= {"✅ accept", "⚠️ review"}
        assert "auto-accepted" in summary
        assert highlights  # source view built

    def test_confidence_is_a_number_in_unit_range(self):
        text, schema = core.EXAMPLES[0]
        rows, _, _ = core.run_verify(text, schema, "text-search", 0.8)
        for r in rows:
            assert 0.0 <= float(r[2]) <= 1.0

    def test_empty_input_is_handled(self):
        rows, highlights, summary = core.run_verify("", core.DEFAULT_SCHEMA)
        assert rows == []
        assert summary == "" and highlights  # a gentle prompt, not a crash

    def test_bad_schema_reports_not_raises(self):
        rows, _highlights, summary = core.run_verify("Total: 5", "{not json")
        assert rows == [] and summary.startswith("⚠️")


class TestResolveAdapter:
    def test_local_adapter_default(self):
        # text-search resolves without a key
        assert core.resolve_adapter("text-search") is not None

    def test_api_vlm_requires_key(self):
        with pytest.raises(ValueError):
            core.resolve_adapter("api-vlm", api_key=None)
