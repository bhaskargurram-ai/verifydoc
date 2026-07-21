"""Offline tests for the FUNSD loader helpers (fixture annotation, no network)."""

import pytest

from benchmark.datasets.funsd import bench_from_annotation, slugify

WIDTH, HEIGHT = 800, 1000

ANNOTATION = {
    "form": [
        {
            "id": 0,
            "label": "question",
            "text": "DATE:",
            "words": [{"text": "DATE:", "box": [50, 100, 120, 130]}],
            "linking": [[0, 1]],
        },
        {
            "id": 1,
            "label": "answer",
            "text": "9/30/99",
            "words": [{"text": "9/30/99", "box": [130, 100, 220, 130]}],
            "linking": [[0, 1]],
        },
        {
            "id": 2,
            "label": "question",
            "text": "TO:",
            "words": [{"text": "TO:", "box": [50, 200, 90, 230]}],
            "linking": [[2, 3], [2, 4]],
        },
        {
            "id": 3,
            "label": "answer",
            "text": "John",
            "words": [{"text": "John", "box": [100, 200, 160, 230]}],
            "linking": [[2, 3]],
        },
        {
            "id": 4,
            "label": "answer",
            "text": "Smith",
            "words": [{"text": "Smith", "box": [170, 200, 240, 230]}],
            "linking": [[2, 4]],
        },
        {
            "id": 5,
            "label": "header",
            "text": "MEMO",
            "words": [{"text": "MEMO", "box": [300, 30, 400, 60]}],
            "linking": [],
        },
    ]
}


class TestSlugify:
    def test_basic(self):
        assert slugify("DATE:") == "date"
        assert slugify("Total Amount ($)") == "total_amount"
        assert slugify("???") == "field"


class TestBenchFromAnnotation:
    def test_gold_fields_from_links(self):
        item = bench_from_annotation("f1", ANNOTATION, WIDTH, HEIGHT)
        golds = {g.path: g for g in item.golds}
        assert set(golds) == {"date", "to"}
        assert golds["to"].value == "John Smith"  # multi-answer concatenation
        assert golds["date"].scoring == "semantic"  # 9/30/99 doesn't parse numerically
        assert golds["to"].scoring == "semantic"

    def test_gold_box_is_answer_union(self):
        item = bench_from_annotation("f1", ANNOTATION, WIDTH, HEIGHT)
        box = {g.path: g for g in item.golds}["to"].gold_box
        assert box.bbox == pytest.approx((100 / WIDTH, 200 / HEIGHT, 240 / WIDTH, 230 / HEIGHT))

    def test_text_layer_includes_all_entries(self):
        item = bench_from_annotation("f1", ANNOTATION, WIDTH, HEIGHT)
        text = item.doc.pages[0].text
        assert "DATE: " not in text  # entries are separate lines
        assert "MEMO" in text and "9/30/99" in text

    def test_schema_matches_golds(self):
        item = bench_from_annotation("f1", ANNOTATION, WIDTH, HEIGHT)
        assert {leaf.path for leaf in item.schema.leaves} == {g.path for g in item.golds}

    def test_empty_annotation_returns_none(self):
        assert bench_from_annotation("f2", {"form": []}, WIDTH, HEIGHT) is None
