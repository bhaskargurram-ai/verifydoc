"""Offline tests for CORD loader helpers (fixture rows; no network, rule #5)."""

import pytest

from benchmark.datasets.cord import (
    CORD_SCHEMA_RAW,
    document_from_valid_lines,
    golds_from_gt_parse,
)
from verifydoc.types import Schema

# One fixture row shaped exactly like a streamed CORD record (values from a
# real validation receipt, truncated).
WIDTH, HEIGHT = 864, 1296
VALID_LINE = [
    {
        "words": [
            {
                "text": "REAL",
                "quad": {
                    "x1": 100,
                    "y1": 100,
                    "x2": 180,
                    "y2": 100,
                    "x3": 180,
                    "y3": 130,
                    "x4": 100,
                    "y4": 130,
                },
            },
            {
                "text": "GANACHE",
                "quad": {
                    "x1": 190,
                    "y1": 100,
                    "x2": 330,
                    "y2": 100,
                    "x3": 330,
                    "y3": 130,
                    "x4": 190,
                    "y4": 130,
                },
            },
        ]
    },
    {
        "words": [
            {
                "text": "TOTAL",
                "quad": {
                    "x1": 100,
                    "y1": 550,
                    "x2": 200,
                    "y2": 550,
                    "x3": 200,
                    "y3": 585,
                    "x4": 100,
                    "y4": 585,
                },
            },
            {
                "text": "45,500",
                "quad": {
                    "x1": 600,
                    "y1": 556,
                    "x2": 700,
                    "y2": 556,
                    "x3": 700,
                    "y3": 586,
                    "x4": 600,
                    "y4": 586,
                },
            },
        ]
    },
]
GT_PARSE = {
    "menu": [{"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"}],
    "total": {"total_price": "45,500"},
}


class TestDocumentFromValidLines:
    def test_real_text_layer(self):
        doc = document_from_valid_lines("r1", VALID_LINE, WIDTH, HEIGHT)
        assert doc.pages[0].text == "REAL GANACHE\nTOTAL 45,500"
        assert len(doc.pages[0].words) == 4

    def test_quads_normalized(self):
        doc = document_from_valid_lines("r1", VALID_LINE, WIDTH, HEIGHT)
        total_word = next(w for w in doc.pages[0].words if w.text == "45,500")
        x0, y0, x1, y1 = total_word.bbox
        assert x0 == pytest.approx(600 / WIDTH)
        assert y1 == pytest.approx(586 / HEIGHT)
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1

    def test_degenerate_words_dropped(self):
        bad = [
            {
                "words": [
                    {
                        "text": "x",
                        "quad": {k: 10 for k in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")},
                    }
                ]
            }
        ]
        doc = document_from_valid_lines("r2", bad, WIDTH, HEIGHT)
        assert doc.pages[0].words == []


class TestGoldsFromGtParse:
    def test_paths_scoring_and_boxes(self):
        schema = Schema.from_json_schema(CORD_SCHEMA_RAW, name="cord")
        doc = document_from_valid_lines("r1", VALID_LINE, WIDTH, HEIGHT)
        golds = {g.path: g for g in golds_from_gt_parse(GT_PARSE, schema, doc)}
        assert set(golds) == {"menu[0].nm", "menu[0].cnt", "menu[0].price", "total.total_price"}
        assert golds["menu[0].nm"].scoring == "semantic"
        assert golds["total.total_price"].scoring == "numeric"
        # gold box located on the real page for values present in the text
        box = golds["total.total_price"].gold_box
        assert box is not None and box.support == pytest.approx(1.0)
        assert box.bbox[0] == pytest.approx(600 / WIDTH)
        # value not on the page (cnt "1") -> no gold box, still scored
        assert golds["menu[0].cnt"].gold_box is None or golds["menu[0].cnt"].gold_box.support < 1.0

    def test_indonesian_thousands_prices_score_numerically(self):
        from verifydoc.eval.extraction import parse_number

        assert parse_number("45,500") == 45500.0
        assert parse_number("16,500") == 16500.0
