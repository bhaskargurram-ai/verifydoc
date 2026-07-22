"""VerifyDoc: a trust layer for document -> structured-JSON extraction."""

from verifydoc.pipeline import VerifiedResult, verify, verify_model
from verifydoc.types import Document, FieldGold, FieldPrediction, Grounding, Schema

__version__ = "0.8.0"

__all__ = [
    "Document",
    "FieldGold",
    "FieldPrediction",
    "Grounding",
    "Schema",
    "VerifiedResult",
    "verify",
    "verify_model",
]
