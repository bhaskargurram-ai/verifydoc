"""Attach groundings by locating each predicted value in the document.

For every prediction without a grounding, search the text layer for the
value: best matching word window gives the bbox (union) and the page text
gives the char span. ``support`` is the string similarity of the matched
region to the value — which doubles as the grounding-based confidence signal.
A value that cannot be located anywhere stays ungrounded (support 0), which
is exactly the hallucination smell the policy layer should flag.
"""

from __future__ import annotations

import re

from verifydoc.eval.extraction import normalize_text, normalized_levenshtein_similarity
from verifydoc.types import Document, FieldPrediction, Grounding, Page

MIN_SUPPORT = 0.5

# Grounding-time normalization: a value the model returns as "45500" should
# ground to the page token "45,500" or "$45,500". We strip currency symbols and
# thousands-separator commas (only between digits, so "Smith, John" is safe) so
# numeric provenance matching works on real receipts/forms.
_CURRENCY_RE = re.compile(r"[$€£¥₹]")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")


def _normalize_for_match(text: str) -> str:
    t = normalize_text(text).casefold()
    t = _CURRENCY_RE.sub("", t)
    t = _THOUSANDS_RE.sub("", t)
    return t


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
    target = _normalize_for_match(value)
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


def _window_bbox(window: list) -> tuple[float, float, float, float]:
    return (
        min(w.bbox[0] for w in window),
        min(w.bbox[1] for w in window),
        max(w.bbox[2] for w in window),
        max(w.bbox[3] for w in window),
    )


def _best_window(target: str, page: Page) -> tuple[tuple[float, float, float, float], float] | None:
    """Best contiguous word window by string similarity to the target.

    Fast path: correctly extracted values match some window verbatim, so an
    exact normalized comparison runs first and returns support 1.0 without
    any edit-distance work. The fuzzy scan only runs for values that do not
    appear verbatim (the interesting, possibly-hallucinated ones), and prunes
    windows whose length alone caps similarity below the current best.
    """
    words = page.words
    if not words:
        return None
    n_target_tokens = max(1, len(target.split()))
    max_width = n_target_tokens + 1
    normalized = [_normalize_for_match(w.text) for w in words]

    for start in range(len(words)):
        joined = normalized[start]
        for width in range(1, min(max_width, len(words) - start) + 1):
            if width > 1:
                joined = f"{joined} {normalized[start + width - 1]}"
            if joined == target:
                return _window_bbox(words[start : start + width]), 1.0

    # fuzzy scan only over candidate starts: windows anchored at a word that
    # exactly matches one of the target's tokens (a partially-corrupted value
    # keeps most tokens intact; a pure hallucination gets no anchors at all,
    # which is the correct fast "ungrounded" answer)
    target_tokens = target.split()
    if len(target_tokens) > _LONG_TARGET_TOKENS:
        return _best_window_by_tokens(target_tokens, words, normalized)
    if len(target_tokens) == 1:
        starts: list[int] = list(range(len(words)))
    else:
        token_set = set(target_tokens)
        anchor_starts = {
            max(0, i - offset)
            for i, ntext in enumerate(normalized)
            if ntext in token_set
            for offset in range(len(target_tokens) + 1)
        }
        starts = sorted(anchor_starts)

    best_score = 0.0
    best_bbox: tuple[float, float, float, float] | None = None
    for start in starts:
        joined = normalized[start]
        for width in range(1, min(max_width, len(words) - start) + 1):
            if width > 1:
                joined = f"{joined} {normalized[start + width - 1]}"
            # similarity <= 1 - |len difference| / max(len): prune hopeless windows
            len_cap = 1.0 - abs(len(joined) - len(target)) / max(len(joined), len(target), 1)
            if len_cap <= best_score:
                continue
            score = normalized_levenshtein_similarity(joined, target)
            if score > best_score:
                best_score = score
                best_bbox = _window_bbox(words[start : start + width])
    if best_bbox is None:
        return None
    return best_bbox, best_score


_LONG_TARGET_TOKENS = 12


def _best_window_by_tokens(
    target_tokens: list[str], words: list, normalized: list[str]
) -> tuple[tuple[float, float, float, float], float] | None:
    """Paragraph-length values: score fixed-width windows by token overlap.

    # DECISION: char-level edit distance is O(len^2) and blows up on
    # paragraph answers (FUNSD has 40+-token values); for targets longer than
    # _LONG_TARGET_TOKENS we slide a window of exactly len(target) tokens and
    # score 2*|multiset intersection| / (|window| + |target|) — same [0,1]
    # support semantics, linear cost, pinned by tests.
    """
    from collections import Counter

    width = len(target_tokens)
    if len(words) < 1:
        return None
    target_count = Counter(target_tokens)
    token_set = set(target_tokens)
    starts = sorted(
        {
            max(0, i - offset)
            for i, token in enumerate(normalized)
            if token in token_set
            for offset in (0, width // 2, width - 1)
        }
    )
    best_score, best_bbox = 0.0, None
    for start in starts:
        window = normalized[start : start + width]
        overlap = sum((Counter(window) & target_count).values())
        score = 2.0 * overlap / (len(window) + width)
        if score > best_score:
            best_score = score
            best_bbox = _window_bbox(words[start : start + len(window)])
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
