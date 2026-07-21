"""Mock adapter: canned or noisy predictions for tests and the synthetic benchmark.

No network, no model. Two modes:
- canned: a mapping of doc_id -> predictions, returned verbatim;
- noisy: given gold fields per doc, corrupt/omit/hallucinate at seeded rates
  to simulate a realistic extractor for the benchmark harness.
"""

from __future__ import annotations

import math
import random

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldGold, FieldPrediction, Schema


class MockAdapter(ExtractorAdapter):
    name = "mock"

    def __init__(
        self,
        canned: dict[str, list[FieldPrediction]] | None = None,
        gold: dict[str, list[FieldGold]] | None = None,
        error_rate: float = 0.15,
        omit_rate: float = 0.05,
        hallucinate_rate: float = 0.05,
        seed: int = 0,
    ) -> None:
        self._canned = canned or {}
        self._gold = gold or {}
        self.error_rate = error_rate
        self.omit_rate = omit_rate
        self.hallucinate_rate = hallucinate_rate
        self._rng = random.Random(seed)

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        if doc.doc_id in self._canned:
            return [p.model_copy(deep=True) for p in self._canned[doc.doc_id]]
        if doc.doc_id in self._gold:
            return self._noisy(self._gold[doc.doc_id])
        return []

    def _noisy(self, golds: list[FieldGold]) -> list[FieldPrediction]:
        rng = self._rng
        preds: list[FieldPrediction] = []
        for gold in golds:
            roll = rng.random()
            if roll < self.omit_rate:
                continue
            correct = roll >= self.omit_rate + self.error_rate
            value = gold.value if correct else self._corrupt(gold)
            preds.append(
                FieldPrediction(
                    path=gold.path, value=value, confidence=0.5, meta=self._signals(correct)
                )
            )
        if golds and rng.random() < self.hallucinate_rate:
            preds.append(
                FieldPrediction(
                    path=f"spurious_{rng.randrange(1000)}",
                    value="ghost",
                    confidence=0.5,
                    meta=self._signals(False),
                )
            )
        return preds

    def _signals(self, correct: bool) -> dict[str, object]:
        """Simulated raw signals with realistic pathologies: verbalized is
        inflated regardless of correctness (the RLHF failure mode); token
        scores carry weak-but-real signal."""
        rng = self._rng
        verbalized = min(1.0, max(0.0, 0.9 + rng.uniform(-0.05, 0.08)))
        base = 0.97 if correct else 0.90
        logprobs = [math.log(min(1.0, max(0.05, base + rng.gauss(0.0, 0.03)))) for _ in range(3)]
        return {"verbalized_confidence": verbalized, "token_logprobs": logprobs}

    def _corrupt(self, gold: FieldGold) -> object:
        """Silently-wrong value: digit swaps for numbers, char noise for text."""
        rng = self._rng
        text = str(gold.value)
        if gold.scoring == "numeric":
            digits = [ch for ch in text if ch.isdigit()]
            if len(digits) >= 2:
                chars = list(text)
                positions = [i for i, ch in enumerate(chars) if ch.isdigit()]
                i, j = rng.sample(positions, 2)
                chars[i], chars[j] = chars[j], chars[i]
                return "".join(chars)
            return text + "0"
        if len(text) >= 2:
            i = rng.randrange(len(text) - 1)
            return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
        return text + "x"
