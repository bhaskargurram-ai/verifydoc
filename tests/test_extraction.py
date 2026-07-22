"""Numeric regression tests for eval/extraction.py against hand-computed values."""

import pytest

from verifydoc.eval.extraction import (
    GridCell,
    anls,
    cer,
    exact_match,
    grits,
    levenshtein,
    normalized_levenshtein_similarity,
    parse_number,
    score_fields,
    teds,
    value_correct,
    wer,
)
from verifydoc.types import FieldGold, FieldPrediction


class TestStringDistances:
    def test_levenshtein_known(self):
        assert levenshtein("kitten", "sitting") == 3
        assert levenshtein("", "abc") == 3
        assert levenshtein("abc", "abc") == 0

    def test_normalized_similarity(self):
        assert normalized_levenshtein_similarity("abc", "abd") == pytest.approx(2 / 3)
        assert normalized_levenshtein_similarity("", "") == 1.0

    def test_cer(self):
        assert cer("helo", "hello") == pytest.approx(1 / 5)
        assert cer("", "") == 0.0
        assert cer("xx", "") == 2.0

    def test_wer(self):
        assert wer("the cat mat", "the cat sat") == pytest.approx(1 / 3)
        assert wer("", "the cat") == 1.0

    def test_anls_hand_computed(self):
        # pair 1: lev("hello","helo")=1, max len 5 -> NLS 0.8 (>= 0.5, kept)
        # pair 2: lev("xyz","abc")=3,   max len 3 -> NLS 0.0 (< 0.5, zeroed)
        assert anls(["hello", "xyz"], ["helo", "abc"]) == pytest.approx(0.4)

    def test_anls_length_mismatch(self):
        with pytest.raises(ValueError):
            anls(["a"], [])


class TestValueScoring:
    def test_parse_number(self):
        assert parse_number("$1,234.50") == pytest.approx(1234.50)
        assert parse_number("42,5") == pytest.approx(42.5)
        assert parse_number(7) == 7.0
        assert parse_number("n/a") is None

    def test_parse_number_strips_currency_codes(self):
        assert parse_number("RM 45.50") == pytest.approx(45.50)
        assert parse_number("Rp 45.500") == pytest.approx(45.500)
        assert parse_number("45.50 USD") == pytest.approx(45.50)
        assert parse_number("USD1,234.50") == pytest.approx(1234.50)

    def test_numeric_rule(self):
        gold = FieldGold(path="total", value=42.50, scoring="numeric", numeric_tol=0.01)
        assert value_correct("$42.50", gold)
        assert value_correct(42.505, gold)
        assert not value_correct(45.20, gold)

    def test_exact_rule(self):
        gold = FieldGold(path="id", value="INV-001", scoring="exact")
        assert value_correct(" inv-001 ", gold)
        assert not value_correct("INV-002", gold)

    def test_semantic_rule(self):
        gold = FieldGold(path="vendor", value="ACME Corp", scoring="semantic")
        # lev("acme corp","acme corporation") = 7, max 16 -> sim 0.5625 >= 0.5
        assert value_correct("Acme Corporation", gold)
        assert not value_correct("Globex", gold)

    def test_none_handling(self):
        gold = FieldGold(path="x", value="v")
        assert not value_correct(None, gold)
        assert value_correct(None, FieldGold(path="x", value=None))

    def test_exact_match_predicate(self):
        assert exact_match(" A B ", "a  b")
        assert not exact_match("a", "b")


class TestScoreFields:
    GOLDS = [
        FieldGold(path="total", value=42.50, scoring="numeric", numeric_tol=0.01),
        FieldGold(path="vendor", value="ACME Corp", scoring="semantic"),
        FieldGold(path="invoice_id", value="INV-001", scoring="exact"),
        FieldGold(path="date", value="2024-01-01", scoring="exact"),
    ]
    PREDS = [
        FieldPrediction(path="total", value=45.20, confidence=0.9),  # silently wrong
        FieldPrediction(path="vendor", value="Acme Corporation", confidence=0.8),  # ok
        FieldPrediction(path="invoice_id", value="INV-001", confidence=0.95),  # ok
        FieldPrediction(path="tax_id", value="XX-99", confidence=0.7),  # hallucinated
        # "date" omitted
    ]

    def test_hand_computed_report(self):
        report = score_fields(self.PREDS, self.GOLDS)
        assert report.n_gold == 4
        assert report.n_predicted == 4
        assert report.n_correct == 2
        assert report.precision == pytest.approx(0.5)
        assert report.recall == pytest.approx(0.5)
        assert report.f1 == pytest.approx(0.5)
        assert report.omission_rate == pytest.approx(0.25)
        assert report.hallucination_rate == pytest.approx(0.25)
        assert report.exact_match_rate == pytest.approx(0.25)
        assert report.omitted_paths == ["date"]
        assert report.hallucinated_paths == ["tax_id"]

    def test_field_scores_feed_selective_layer(self):
        report = score_fields(self.PREDS, self.GOLDS)
        by_path = {s.path: s for s in report.field_scores}
        assert not by_path["total"].correct and by_path["total"].confidence == 0.9
        assert by_path["tax_id"].status == "hallucinated"
        assert by_path["vendor"].correct

    def test_none_prediction_is_omission(self):
        report = score_fields(
            [FieldPrediction(path="date", value=None, confidence=0.9)], self.GOLDS
        )
        assert "date" in report.omitted_paths
        assert report.n_predicted == 0

    def test_duplicate_gold_paths_rejected(self):
        with pytest.raises(ValueError):
            score_fields([], [FieldGold(path="a"), FieldGold(path="a")])

    def test_empty_inputs(self):
        report = score_fields([], [])
        assert report.f1 == 0.0 and report.omission_rate == 0.0


TABLE = "<table><tr><td>a</td><td>b</td></tr></table>"  # 4 nodes


class TestTEDS:
    def test_identical(self):
        assert teds(TABLE, TABLE) == pytest.approx(1.0)

    def test_one_cell_text_changed(self):
        pred = "<table><tr><td>a</td><td>c</td></tr></table>"
        # rename cost = 1 - sim("c","b") = 1; max tree size 4 -> 1 - 1/4
        assert teds(pred, TABLE) == pytest.approx(0.75)
        assert teds(pred, TABLE, struct_only=True) == pytest.approx(1.0)

    def test_missing_cell(self):
        pred = "<table><tr><td>a</td></tr></table>"  # 3 nodes; one deletion
        assert teds(pred, TABLE) == pytest.approx(0.75)

    def test_colspan_mismatch_is_structural(self):
        pred = '<table><tr><td colspan="2">a</td><td>b</td></tr></table>'
        assert teds(pred, TABLE, struct_only=True) == pytest.approx(0.75)

    def test_partial_text_credit(self):
        pred = "<table><tr><td>a</td><td>bx</td></tr></table>"
        # rename cost = 1 - (1 - lev("bx","b")/2) = 0.5 -> 1 - 0.5/4
        assert teds(pred, TABLE) == pytest.approx(0.875)

    def test_no_table_raises(self):
        with pytest.raises(ValueError):
            teds("<div>no</div>", TABLE)


def _grid(rows):
    return [[GridCell(text=t) for t in row] for row in rows]


class TestGriTS:
    def test_identical_con(self):
        g = _grid([["a", "b"], ["c", "d"]])
        assert grits(g, g, "con") == pytest.approx((1.0, 1.0, 1.0))

    def test_one_cell_wrong_con(self):
        pred = _grid([["a", "b"], ["c", "x"]])
        gold = _grid([["a", "b"], ["c", "d"]])
        # LCS("x","d")=0 -> total sim 3 of 4 cells
        p, r, f = grits(pred, gold, "con")
        assert (p, r, f) == pytest.approx((0.75, 0.75, 0.75))

    def test_partial_text_credit_con(self):
        pred = _grid([["ab"]])
        gold = _grid([["abcd"]])
        # 2*LCS(2)/(2+4) = 2/3
        p, r, f = grits(pred, gold, "con")
        assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)

    def test_missing_row_recall(self):
        pred = _grid([["a", "b"]])
        gold = _grid([["a", "b"], ["c", "d"]])
        p, r, f = grits(pred, gold, "con")
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(0.5)
        assert f == pytest.approx(2 / 3)

    def test_topology_span_mismatch(self):
        pred = [[GridCell(rowspan=2)]]
        gold = [[GridCell(rowspan=1)]]
        # IoU of extents = 1 / (2 + 1 - 1) = 0.5
        p, r, f = grits(pred, gold, "top")
        assert p == pytest.approx(0.5)

    def test_location_variant(self):
        pred = [[GridCell(bbox=(0, 0, 1, 1))]]
        gold = [[GridCell(bbox=(0, 0, 1, 1))]]
        assert grits(pred, gold, "loc") == pytest.approx((1.0, 1.0, 1.0))

    def test_empty(self):
        assert grits([], _grid([["a"]]), "con") == (0.0, 0.0, 0.0)
