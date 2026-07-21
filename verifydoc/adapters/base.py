"""The ExtractorAdapter interface: extract(doc, schema) -> list[FieldPrediction]."""

from __future__ import annotations

from abc import ABC, abstractmethod

from verifydoc.types import Document, FieldPrediction, Schema


class ExtractorAdapter(ABC):
    """Wraps one extraction model behind a model-agnostic interface.

    Adapters emit raw predictions (values + whatever raw signals they have in
    ``meta``); confidence, calibration, grounding, and abstention are applied
    by later stages, never inside an adapter.
    """

    name: str = "adapter"

    @abstractmethod
    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        """Extract one prediction per schema leaf found in the document."""

    def extract_samples(
        self, doc: Document, schema: Schema, k: int = 1
    ) -> list[list[FieldPrediction]]:
        """k independent runs for self-consistency; override when the model
        supports cheaper native sampling (e.g. temperature > 0 decoding)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return [self.extract(doc, schema) for _ in range(k)]
