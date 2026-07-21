"""Attach groundings by locating each predicted value in the document.

For every prediction without a grounding, search the text layer for the
value: best matching word window gives the bbox (union) and the page text
gives the char span. ``support`` is the string similarity of the matched
region to the value — which doubles as the grounding-based confidence signal.
A value that cannot be located anywhere stays ungrounded (support 0), which
is exactly the hallucination smell the policy layer should flag.
"""

from __future__ import annotations

from verifydoc.eval.extraction import normalize_text, normalized_levenshtein_similarity
from verifydoc.types import Document, FieldPrediction, Grounding, Page

MIN_SUPPORT = 0.5


def ground_predictions(
    predictions: list[FieldPrediction], doc: Document, min_support: float = MIN_SUPPORT
) -> list[FieldPrediction]:
    """Return copies with grounding attached where the value can be located."""
    out = []
    for pred in predictions:
        if pred.grounding is not None or pred.value is None:
            out.append(pred)
            continue
        grounding = _locate(str(pred.value), doc, min_support)
        out.append(pred if grounding is None else pred.model_copy(update={"grounding": grounding}))
    return out


def _locate(value: str, doc: Document, min_support: float) -> Grounding | None:
    target = normalize_text(value).casefold()
    if not target:
        return None
    best: Grounding | None = None
    best_support = min_support
    for page in doc.pages:
        found = _best_window(target, page)
        if found is not None and found[1] >= best_support:
            bbox, support = found
            best = Grounding(
                page=page.page,
                bbox=bbox,
                char_span=_char_span(target, page),
                support=support,
            )
            best_support = support
    return best


def _best_window(target: str, page: Page) -> tuple[tuple[float, float, float, float], float] | None:
    """Best contiguous word window by string similarity to the target."""
    words = page.words
    if not words:
        return None
    n_target_tokens = max(1, len(target.split()))
    best_score = 0.0
    best_bbox: tuple[float, float, float, float] | None = None
    for start in range(len(words)):
        for width in range(1, min(n_target_tokens + 2, len(words) - start + 1)):
            window = words[start : start + width]
            text = normalize_text(" ".join(w.text for w in window)).casefold()
            score = normalized_levenshtein_similarity(text, target)
            if score > best_score:
                bbox = (
                    min(w.bbox[0] for w in window),
                    min(w.bbox[1] for w in window),
                    max(w.bbox[2] for w in window),
                    max(w.bbox[3] for w in window),
                )
                best_score, best_bbox = score, bbox
                if score >= 1.0:
                    return bbox, score
    if best_bbox is None:
        return None
    return best_bbox, best_score


def _char_span(target: str, page: Page) -> tuple[int, int] | None:
    if not page.text:
        return None
    idx = page.text.casefold().find(target)
    if idx == -1:
        return None
    return (idx, idx + len(target))
