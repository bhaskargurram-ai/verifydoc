"""Tests for the VerifyDocBench leaderboard (offline, fixture summaries)."""

import json

from verifydoc.eval.leaderboard import leaderboard_row, leaderboard_table, load_summaries

SUMMARIES = [
    {
        "dataset": "cord",
        "extractor": "api-vlm",
        "extraction": {"f1": 0.47, "exact_match": 0.25, "hallucination_rate": 0.47},
        "grounding": {"grounding_gap": 0.28},
        "best_coverage": {"auroc": 0.845},
        "best_ece": {"ece": 0.013},
    },
    {
        "dataset": "cord",
        "extractor": "rapidocr",
        "extraction": {"f1": 0.28, "exact_match": 0.19, "hallucination_rate": 0.0},
        "grounding": {"grounding_gap": 0.83},
        "best_coverage": {"auroc": 0.89},
        "best_ece": {"ece": 0.04},
    },
]


class TestLeaderboardRow:
    def test_pulls_trust_dimensions(self):
        row = leaderboard_row(SUMMARIES[0])
        assert row["extractor"] == "api-vlm"
        assert row["halluc"] == 0.47 and row["best_AUROC"] == 0.845 and row["grounding_gap"] == 0.28

    def test_missing_keys_default_to_zero(self):
        row = leaderboard_row({"extractor": "mock"})
        assert row["F1"] == 0.0 and row["best_AUROC"] == 0.0


class TestLeaderboardTable:
    def test_ranked_by_auroc_desc(self):
        table = leaderboard_table(SUMMARIES)
        assert "VerifyDocBench leaderboard" in table
        # rapidocr (AUROC 0.89) ranks above api-vlm (0.845)
        assert table.index("rapidocr") < table.index("api-vlm")
        assert "grounding_gap" in table and "halluc" in table

    def test_empty(self):
        assert "leaderboard" in leaderboard_table([]).lower()


class TestLoadSummaries:
    def test_loads_and_skips_bad(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(SUMMARIES[0]))
        (tmp_path / "bad.json").write_text("{not json")
        loaded = load_summaries(
            [tmp_path / "a.json", tmp_path / "bad.json", tmp_path / "nope.json"]
        )
        assert len(loaded) == 1 and loaded[0]["extractor"] == "api-vlm"
