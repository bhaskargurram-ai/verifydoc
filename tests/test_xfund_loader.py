"""Offline tests for the XFUND loader (fixture annotations, no network)."""

from benchmark.datasets.funsd import bench_from_annotation

WIDTH, HEIGHT = 1000, 1000

# XFUND uses the same form structure as FUNSD: a list of entries with
# id/text/box/label/linking/words. This fixture mirrors one QA pair.
FORM = [
    {
        "id": 0,
        "label": "question",
        "text": "Datum:",
        "box": [50, 100, 150, 130],
        "words": [{"text": "Datum:", "box": [50, 100, 150, 130]}],
        "linking": [[0, 1]],
    },
    {
        "id": 1,
        "label": "answer",
        "text": "15.03.2024",
        "box": [160, 100, 300, 130],
        "words": [{"text": "15.03.2024", "box": [160, 100, 300, 130]}],
        "linking": [[0, 1]],
    },
]


class TestXFUNDViaFUNSDHelper:
    """XFUND reuses FUNSD's bench_from_annotation, so we verify the bridge."""

    def test_qa_pair_becomes_gold_field(self):
        item = bench_from_annotation("xfund-de-val-0001", {"form": FORM}, WIDTH, HEIGHT)
        assert item is not None
        assert len(item.golds) == 1
        gold = item.golds[0]
        assert gold.value == "15.03.2024"
        assert gold.path == "datum"
        assert gold.gold_box is not None
        assert gold.gold_box.page == 0

    def test_text_layer_built_from_words(self):
        item = bench_from_annotation("xfund-de-val-0001", {"form": FORM}, WIDTH, HEIGHT)
        assert item is not None
        text = item.doc.pages[0].text or ""
        assert "Datum:" in text
        assert "15.03.2024" in text
        assert len(item.doc.pages[0].words) == 2

    def test_empty_form_returns_none(self):
        item = bench_from_annotation("xfund-de-val-empty", {"form": []}, WIDTH, HEIGHT)
        assert item is None
