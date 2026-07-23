"""Entailment-based grounding verification (grounding-sweep P4).

Lexical grounding ``support`` only asks *is the value present in the source
span?* — it cannot tell ``Total: 100`` grounded onto a line that reads
``Subtotal: 100`` (value present, proposition false). This module verifies that
the grounded span **entails** the proposition ``"{field} = {value}"`` with a
natural-language-inference scorer, and folds that entailment probability into
``grounding.support`` so semantically-wrong-but-lexically-present values lose
confidence and route to review.

The NLI scorer is pluggable behind :class:`EntailmentScorer`. The zero-dependency
default (:class:`LexicalEntailmentScorer`) degrades to token overlap — it does
*not* deliver the semantic check on its own; plug a real MNLI cross-encoder
(:class:`CrossEncoderEntailmentScorer`) to catch the failure mode above. Model
code stays out of ``adapters/``; this is a confidence signal, not an extractor.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from verifydoc.eval.extraction import normalize_text
from verifydoc.types import FieldPrediction


@runtime_checkable
class EntailmentScorer(Protocol):
    """Scores P(premise entails hypothesis) in ``[0, 1]``."""

    def score(self, premise: str, hypothesis: str) -> float: ...


class LexicalEntailmentScorer:
    """Zero-dependency fallback: fraction of hypothesis tokens present in the
    premise (casefolded). Degrades to lexical overlap — a stand-in until a real
    NLI cross-encoder is plugged in; documented as such so results stay honest.
    """

    def score(self, premise: str, hypothesis: str) -> float:
        prem = set(normalize_text(premise).casefold().split())
        hyp = normalize_text(hypothesis).casefold().split()
        if not hyp:
            return 0.0
        return sum(1 for t in hyp if t in prem) / len(hyp)


class CrossEncoderEntailmentScorer:
    """Wrap a HuggingFace MNLI cross-encoder as an :class:`EntailmentScorer`.

    Lazily imports ``sentence_transformers`` only when constructed, so importing
    this module never pulls in torch. ``entail_index`` selects the entailment
    logit for the chosen model's label order (deberta-v3 MNLI → 1); the three
    logits are softmaxed to a probability.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        *,
        entail_index: int = 1,
        model: Any = None,
    ) -> None:
        self._entail_index = entail_index
        if model is not None:
            self._model = model
        else:  # pragma: no cover - requires the optional heavy dependency
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)

    def score(self, premise: str, hypothesis: str) -> float:
        import math

        logits = list(self._model.predict([(premise, hypothesis)])[0])
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        return exps[self._entail_index] / sum(exps)


_DEFAULT_TEMPLATE = "{field} is {value}"


def _hypothesis(pred: FieldPrediction, template: str) -> str:
    field = pred.path.split(".")[-1].replace("_", " ")
    return template.format(field=field, value=pred.value, path=pred.path)


def _premise(pred: FieldPrediction, source_text: str) -> str | None:
    """Grounded span text via ``char_span``; whole ``source_text`` if grounded
    without a span; ``None`` if there is nothing to verify against."""
    g = pred.grounding
    if g is None or not source_text:
        return None
    if g.char_span is not None:
        start, end = g.char_span
        return source_text[start:end]
    return source_text


_COMBINERS = {
    "min": lambda old, e: min(old, e),
    "product": lambda old, e: old * e,
    "mean": lambda old, e: (old + e) / 2,
    "replace": lambda old, e: e,
}


def entailment_support(
    pred: FieldPrediction,
    source_text: str,
    scorer: EntailmentScorer,
    *,
    template: str = _DEFAULT_TEMPLATE,
) -> float | None:
    """P(grounded span entails ``"{field} = {value}"``), or ``None`` if the
    prediction is ungrounded / has no source to check against."""
    premise = _premise(pred, source_text)
    if premise is None:
        return None
    return scorer.score(premise, _hypothesis(pred, template))


def apply_entailment_grounding(
    predictions: list[FieldPrediction],
    source_text: str,
    scorer: EntailmentScorer | None = None,
    *,
    template: str = _DEFAULT_TEMPLATE,
    combine: str = "min",
) -> list[FieldPrediction]:
    """Return copies whose ``grounding.support`` folds in span entailment.

    ``combine`` decides how entailment ``e`` meets the existing (lexical)
    support ``s``: ``"min"`` (default — a field must be both located *and*
    entailed), ``"product"``, ``"mean"``, or ``"replace"``. Ungrounded fields
    and fields with no source text are returned unchanged.
    """
    if combine not in _COMBINERS:
        raise ValueError(f"unknown combine {combine!r}; choose from {sorted(_COMBINERS)}")
    scorer = scorer or LexicalEntailmentScorer()
    merge = _COMBINERS[combine]

    out: list[FieldPrediction] = []
    for pred in predictions:
        e = entailment_support(pred, source_text, scorer, template=template)
        if e is None or pred.grounding is None:
            out.append(pred)
            continue
        new_support = max(0.0, min(1.0, merge(pred.grounding.support, e)))
        new_grounding = pred.grounding.model_copy(update={"support": new_support})
        out.append(pred.model_copy(update={"grounding": new_grounding}))
    return out
