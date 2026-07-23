"""Tests for verify_batch and the observability helpers (offline)."""

import json
import logging

from verifydoc import verify, verify_batch
from verifydoc.ingest import document_from_text
from verifydoc.observability import log_verification, timed, verification_event

TEXT = "Corner Cafe\nDate: 2024-05-01\nTotal: 7.70\n"
SCHEMA = {"type": "object", "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}}}


def _doc(doc_id="cafe"):
    return document_from_text(doc_id, [TEXT])


class TestVerifyBatch:
    def test_returns_one_result_per_source_in_order(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        results = verify_batch(docs, schema=SCHEMA)
        assert [r.doc_id for r in results] == ["a", "b", "c"]

    def test_matches_single_verify(self):
        single = verify(_doc(), schema=SCHEMA)
        (batched,) = verify_batch([_doc()], schema=SCHEMA)
        assert single.to_dict() == batched.to_dict()

    def test_empty_batch(self):
        assert verify_batch([], schema=SCHEMA) == []


class TestObservability:
    def test_verification_event_shape(self):
        event = verification_event(verify(_doc(), schema=SCHEMA))
        assert event["event"] == "verifydoc.verification"
        assert event["n_accepted"] + event["n_review"] == event["n_fields"]
        assert 0.0 <= event["mean_confidence"] <= 1.0
        assert "duration_ms" not in event

    def test_duration_included_when_given(self):
        event = verification_event(verify(_doc(), schema=SCHEMA), duration_s=0.5)
        assert event["duration_ms"] == 500.0

    def test_log_verification_emits_json(self, caplog):
        result = verify(_doc(), schema=SCHEMA)
        with caplog.at_level(logging.INFO, logger="verifydoc"):
            event = log_verification(result)
        assert event["doc_id"] == result.doc_id
        record = json.loads(caplog.records[-1].getMessage())
        assert record["doc_id"] == result.doc_id

    def test_timed_context_manager(self):
        with timed() as t:
            verify(_doc(), schema=SCHEMA)
        assert t.seconds >= 0.0
