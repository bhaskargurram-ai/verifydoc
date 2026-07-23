"""Confidence signals (PROJECT.md §5.G): raw per-field scores before calibration."""

from verifydoc.confidence.combined import combined_confidence
from verifydoc.confidence.consensus import adaptive_consensus, consensus
from verifydoc.confidence.entailment import (
    CrossEncoderEntailmentScorer,
    EntailmentScorer,
    LexicalEntailmentScorer,
    apply_entailment_grounding,
    entailment_support,
)
from verifydoc.confidence.grounding_based import apply_grounding_confidence, grounding_confidence
from verifydoc.confidence.token_prob import apply_token_prob, token_prob_confidence
from verifydoc.confidence.verbalized import apply_verbalized, verbalized_confidence

__all__ = [
    "CrossEncoderEntailmentScorer",
    "EntailmentScorer",
    "LexicalEntailmentScorer",
    "adaptive_consensus",
    "apply_entailment_grounding",
    "apply_grounding_confidence",
    "apply_token_prob",
    "apply_verbalized",
    "combined_confidence",
    "consensus",
    "entailment_support",
    "grounding_confidence",
    "token_prob_confidence",
    "verbalized_confidence",
]
