"""End-to-end tests: ingest -> adapter -> confidence -> grounding -> policy -> CLI.

All offline (golden rule #5): text documents, mock/heuristic/fake-client adapters.
"""

import json

import pytest
from typer.testing import CliRunner

from verifydoc import verify
from verifydoc.adapters import MockAdapter, TextSearchAdapter, get_adapter
from verifydoc.adapters._ocr_common import OCRToken, predictions_from_ocr_tokens
from verifydoc.adapters.api_vlm import APIVLMAdapter
from verifydoc.adapters.docling import DoclingAdapter
from verifydoc.cli import app
from verifydoc.grounding import ground_predictions
from verifydoc.ingest import document_from_text, ingest_path
from verifydoc.policy import apply_policy, threshold_for_target_risk
from verifydoc.types import FieldGold, FieldPrediction, Schema

INVOICE_TEXT = """ACME CORPORATION
Invoice ID: INV-2024-001
Vendor: ACME Corporation
Date: 2024-01-15
Total: $1,234.50
Thank you for your business"""

SCHEMA_RAW = {
    "type": "object",
    "required": ["invoice_id", "total"],
    "properties": {
        "invoice_id": {"type": "string"},
        "vendor": {"type": "string", "x-scoring": "semantic"},
        "date": {"type": "string"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}
SCHEMA = Schema.from_json_schema(SCHEMA_RAW, name="invoice")


def make_doc():
    return document_from_text("inv-001", [INVOICE_TEXT])


class TestIngest:
    def test_document_from_text_layout(self):
        doc = make_doc()
        assert doc.n_pages == 1
        page = doc.pages[0]
        assert "INV-2024-001" in (page.text or "")
        assert any(w.text == "INV-2024-001" for w in page.words)
        for w in page.words:
            x0, y0, x1, y1 = w.bbox
            assert 0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1

    def test_ingest_text_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text(INVOICE_TEXT, encoding="utf-8")
        doc = ingest_path(f)
        assert doc.doc_id == "doc" and doc.n_pages == 1

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ingest_path("nope.txt")


class TestTextSearchAdapter:
    def test_extracts_labeled_fields(self):
        preds = TextSearchAdapter().extract(make_doc(), SCHEMA)
        by_path = {p.path: p.value for p in preds}
        assert by_path["invoice_id"] == "INV-2024-001"
        assert by_path["vendor"] == "ACME Corporation"
        assert by_path["date"] == "2024-01-15"
        assert by_path["total"] == "$1,234.50"

    def test_skips_array_leaves(self):
        schema = Schema.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            }
        )
        assert TextSearchAdapter().extract(make_doc(), schema) == []


class TestMockAdapter:
    GOLD = [
        FieldGold(path="total", value="42.50", scoring="numeric"),
        FieldGold(path="vendor", value="ACME Corp"),
    ]

    def test_canned_mode(self):
        canned = {"inv-001": [FieldPrediction(path="total", value="1")]}
        adapter = MockAdapter(canned=canned)
        assert adapter.extract(make_doc(), SCHEMA)[0].value == "1"

    def test_noisy_mode_seeded(self):
        a = MockAdapter(gold={"inv-001": self.GOLD}, seed=1)
        b = MockAdapter(gold={"inv-001": self.GOLD}, seed=1)
        doc = make_doc()
        assert [p.value for p in a.extract(doc, SCHEMA)] == [
            p.value for p in b.extract(doc, SCHEMA)
        ]

    def test_numeric_corruption_is_digit_swap(self):
        adapter = MockAdapter(gold={"inv-001": self.GOLD}, error_rate=1.0, omit_rate=0.0)
        preds = adapter.extract(make_doc(), SCHEMA)
        wrong_total = next(p.value for p in preds if p.path == "total")
        assert wrong_total != "42.50"
        assert sorted(str(wrong_total)) == sorted("42.50")  # same digits, swapped

    def test_get_adapter_registry(self):
        assert isinstance(get_adapter("mock"), MockAdapter)
        with pytest.raises(ValueError):
            get_adapter("nope")

    def test_extract_samples_validates_k(self):
        with pytest.raises(ValueError):
            TextSearchAdapter().extract_samples(make_doc(), SCHEMA, k=0)


class TestOCRCommon:
    TOKENS = [
        OCRToken(text="Total:", bbox=(0.1, 0.50, 0.2, 0.54), score=0.99),
        OCRToken(text="$42.50", bbox=(0.25, 0.50, 0.35, 0.54), score=0.75),
        OCRToken(text="Invoice", bbox=(0.1, 0.10, 0.2, 0.14), score=0.99),
        OCRToken(text="ID:", bbox=(0.22, 0.10, 0.26, 0.14), score=0.99),
        OCRToken(text="INV-7", bbox=(0.3, 0.10, 0.4, 0.14), score=0.95),
    ]

    def test_line_clustering_and_extraction(self):
        preds = predictions_from_ocr_tokens("d1", self.TOKENS, SCHEMA)
        by_path = {p.path: p for p in preds}
        assert by_path["total"].value == "$42.50"
        assert by_path["invoice_id"].value == "INV-7"

    def test_token_logprobs_attached(self):
        preds = predictions_from_ocr_tokens("d1", self.TOKENS, SCHEMA)
        total = next(p for p in preds if p.path == "total")
        assert total.meta["token_logprobs"] == [pytest.approx(-0.2876820724)]


class TestDoclingAdapter:
    def test_parsed_output_mode(self, tmp_path):
        parsed = tmp_path / "parsed.md"
        parsed.write_text(INVOICE_TEXT, encoding="utf-8")
        preds = DoclingAdapter(parsed_output=parsed).extract(make_doc(), SCHEMA)
        assert {p.path: p.value for p in preds}["invoice_id"] == "INV-2024-001"


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt):
        assert "Schema" in prompt
        return self.payload


class TestAPIVLMAdapter:
    def test_parses_json_with_verbalized_confidence(self):
        payload = json.dumps(
            {
                "invoice_id": {"value": "INV-2024-001", "confidence": 0.95},
                "total": {"value": 1234.5, "confidence": 0.6},
            }
        )
        preds = APIVLMAdapter(client=FakeClient(payload)).extract(make_doc(), SCHEMA)
        by_path = {p.path: p for p in preds}
        assert by_path["invoice_id"].meta["verbalized_confidence"] == 0.95
        assert by_path["total"].value == 1234.5

    def test_garbage_response_yields_nothing(self):
        assert APIVLMAdapter(client=FakeClient("sorry, no")).extract(make_doc(), SCHEMA) == []


class TestGrounder:
    def test_attaches_span_bbox_support(self):
        preds = TextSearchAdapter().extract(make_doc(), SCHEMA)
        grounded = ground_predictions(preds, make_doc())
        inv = next(p for p in grounded if p.path == "invoice_id")
        assert inv.grounding is not None
        assert inv.grounding.support == pytest.approx(1.0)
        assert inv.grounding.char_span is not None
        text = make_doc().pages[0].text or ""
        lo, hi = inv.grounding.char_span
        assert text.casefold()[lo:hi] == "inv-2024-001"

    def test_unfindable_value_stays_ungrounded(self):
        pred = FieldPrediction(path="ghost", value="ZZZ-NOT-THERE-999")
        (out,) = ground_predictions([pred], make_doc())
        assert out.grounding is None


class TestPolicy:
    def test_empirical_vs_conformal_thresholds(self):
        conf = [0.9, 0.8, 0.7, 0.6, 0.5]
        corr = [1, 1, 1, 0, 0]
        empirical = threshold_for_target_risk(conf, corr, alpha=0.25, method="empirical")
        conformal = threshold_for_target_risk(conf, corr, alpha=0.25, method="conformal")
        assert empirical.threshold == pytest.approx(0.6)  # risk 1/4 at k=4
        assert conformal.threshold == pytest.approx(0.7)  # finite-sample cushion
        assert conformal.expected_coverage == pytest.approx(0.6)

    def test_apply_policy_decisions(self):
        preds = [
            FieldPrediction(path="a", value="x", confidence=0.9),
            FieldPrediction(path="b", value="y", confidence=0.5),
            FieldPrediction(path="c", value=None, confidence=0.99),  # omission
        ]
        out = apply_policy(preds, threshold=0.8)
        assert [p.decision for p in out] == ["accept", "review", "review"]


class TestVerifyE2E:
    def test_verify_end_to_end(self):
        result = verify(make_doc(), SCHEMA, k=1, threshold=0.8)
        assert result.doc_id == "inv-001"
        assert len(result.fields) == 4
        assert result.n_accepted + result.n_review == 4
        by_path = {f.path: f for f in result.fields}
        # values located verbatim on the page -> grounded, high confidence, accepted
        assert by_path["invoice_id"].decision == "accept"
        assert by_path["invoice_id"].grounding is not None
        payload = result.to_dict()
        assert payload["fields"]["invoice_id"]["value"] == "INV-2024-001"
        assert result.values()["date"] == "2024-01-15"

    def test_verify_with_consensus(self):
        gold = [
            FieldGold(path="invoice_id", value="INV-2024-001"),
            FieldGold(path="total", value="$1,234.50", scoring="numeric"),
        ]
        adapter = MockAdapter(gold={"inv-001": gold}, error_rate=0.3, seed=5)
        result = verify(make_doc(), SCHEMA, adapter=adapter, k=5, threshold=0.6)
        assert result.fields
        for f in result.fields:
            assert 0.0 <= f.confidence <= 1.0
            assert f.decision in ("accept", "review")

    def test_verify_from_file_with_schema_path(self, tmp_path):
        doc_file = tmp_path / "invoice.txt"
        doc_file.write_text(INVOICE_TEXT, encoding="utf-8")
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SCHEMA_RAW), encoding="utf-8")
        result = verify(doc_file, schema_file)
        assert {f.path for f in result.fields} == {"invoice_id", "vendor", "date", "total"}


class TestCLI:
    def test_extract_command(self, tmp_path):
        doc_file = tmp_path / "invoice.txt"
        doc_file.write_text(INVOICE_TEXT, encoding="utf-8")
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(SCHEMA_RAW), encoding="utf-8")
        out_file = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            ["extract", str(doc_file), "--schema", str(schema_file), "--out", str(out_file)],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert payload["fields"]["invoice_id"]["decision"] in ("accept", "review")
        assert "accepted" in result.output

    def test_version_command(self):
        import verifydoc

        result = CliRunner().invoke(app, ["version"])
        assert result.exit_code == 0
        assert verifydoc.__version__ in result.output


class TestGrounderLongValues:
    def test_paragraph_value_grounds_by_token_overlap(self):
        words = " ".join(f"tok{i}" for i in range(30))
        doc = document_from_text("long", [f"HEADER LINE\n{words}\nFOOTER"])
        value = " ".join(f"tok{i}" for i in range(30))  # exact 30-token paragraph
        (out,) = ground_predictions([FieldPrediction(path="p", value=value)], doc)
        assert out.grounding is not None
        assert out.grounding.support == pytest.approx(1.0)

    def test_partially_corrupted_paragraph_partial_support(self):
        words = " ".join(f"tok{i}" for i in range(30))
        doc = document_from_text("long", [words])
        corrupted = " ".join(f"tok{i}" if i % 3 else "zzz" for i in range(30))  # 10 of 30 wrong
        (out,) = ground_predictions(
            [FieldPrediction(path="p", value=corrupted)], doc, min_support=0.5
        )
        assert out.grounding is not None
        assert 0.6 < out.grounding.support < 0.75  # 2*20/(30+30)

    def test_hallucinated_paragraph_stays_ungrounded(self):
        doc = document_from_text("long", ["totally different page content here"])
        ghost = " ".join(f"ghost{i}" for i in range(20))
        (out,) = ground_predictions([FieldPrediction(path="p", value=ghost)], doc)
        assert out.grounding is None


class TestTextSearchAliases:
    def test_alias_label_matches_receipt_style(self):
        doc = document_from_text("r1", ["KOPI KENANGAN\nTOTAL 45,500\nCASH 50,000"])
        schema = Schema.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "total_price": {"type": "number", "x-aliases": ["total"]},
                    "cashprice": {"type": "number", "x-aliases": ["cash", "tunai"]},
                },
            }
        )
        preds = {p.path: p.value for p in TextSearchAdapter().extract(doc, schema)}
        assert preds["total_price"] == "45,500"
        assert preds["cashprice"] == "50,000"
