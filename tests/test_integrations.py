"""Tests for the Instructor/Pydantic + LangChain drop-in integrations (offline)."""

import pytest
from pydantic import BaseModel

from verifydoc.integrations.instructor import verify_instructor_result
from verifydoc.integrations.langchain import VerifiedExtractor

DOC = "ACME CORPORATION\nInvoice ID: INV-2024-001\nTotal: $1,234.50\n"


class Invoice(BaseModel):
    invoice_id: str
    total: float


class TestInstructorIntegration:
    def test_verifies_pydantic_object(self):
        obj = Invoice(invoice_id="INV-2024-001", total=1234.50)
        report = verify_instructor_result(DOC, obj, threshold=0.8)
        by_path = {f.path: f for f in report.fields}
        # invoice_id appears verbatim -> grounded, accepted
        assert by_path["invoice_id"].decision == "accept"
        assert by_path["invoice_id"].grounding is not None

    def test_hallucinated_field_flagged(self):
        # a value not on the page should not ground -> review
        obj = Invoice(invoice_id="INV-9999-XXX", total=1234.50)
        report = verify_instructor_result(DOC, obj, threshold=0.8)
        inv = next(f for f in report.fields if f.path == "invoice_id")
        assert inv.decision == "review"

    def test_rejects_non_pydantic(self):
        with pytest.raises(TypeError):
            verify_instructor_result(DOC, {"invoice_id": "x"})


class TestLangChainIntegration:
    def test_wraps_dict_extractor(self):
        # a fake "chain" that returns a dict
        def fake_chain(text: str) -> dict:
            return {"invoice_id": "INV-2024-001", "total": "$1,234.50"}

        extractor = VerifiedExtractor(fake_chain, Invoice, threshold=0.8)
        result = extractor(DOC)
        assert result.n_accepted + result.n_review == len(result.fields)
        assert {f.path for f in result.fields} == {"invoice_id", "total"}

    def test_wraps_pydantic_returning_extractor(self):
        def fake_chain(text: str) -> Invoice:
            return Invoice(invoice_id="INV-2024-001", total=1234.5)

        extractor = VerifiedExtractor(fake_chain, Invoice)
        result = extractor(DOC)
        assert next(f for f in result.fields if f.path == "invoice_id").grounding is not None

    def test_accepts_dict_schema(self):
        schema = {
            "type": "object",
            "properties": {"invoice_id": {"type": "string"}},
        }
        extractor = VerifiedExtractor(lambda t: {"invoice_id": "INV-2024-001"}, schema)
        assert extractor(DOC).fields[0].path == "invoice_id"
