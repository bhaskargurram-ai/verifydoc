"""Attach groundings by locating each predicted value in the document.

For every prediction without a grounding, search the text layer for the
value: best matching word window gives the bbox (union) and the page text
gives the char span. ``support`` is the string similarity of the matched
region to the value — which doubles as the grounding-based confidence signal.
A value that cannot be located anywhere stays ungrounded (support 0), which
is exactly the hallucination smell the policy layer should flag.
"""

from __future__ import annotations

import math
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
    predictions: list[FieldPrediction],
    doc: Document,
    min_support: float = MIN_SUPPORT,
    penalize_ambiguity: bool = True,
    penalty_mode: str = "uniform",
) -> list[FieldPrediction]:
    """Return copies with grounding attached where the value can be located.

    ``penalize_ambiguity`` (default True) discounts support for values that
    match at multiple places — the right behavior for grounding-as-confidence.
    Set False for pure annotation (locate a known-correct value regardless of
    how many times it appears).

    ``penalty_mode`` sets the down-weighting form when a value matches ``m``
    equally-good locations (ablation knob; see :func:`ambiguity_penalty`):
    ``"uniform"`` (default) uses ``1/m`` — the posterior P(true source) under a
    uniform prior over the ``m`` matches; ``"sqrt"`` (``1/√m``) and ``"log"``
    (``1/(1+ ln m)``) are softer; ``"none"`` disables it.
    """
    mode = "none" if not penalize_ambiguity else penalty_mode
    out = []
    for pred in predictions:
        if pred.grounding is not None or pred.value is None:
            out.append(pred)
            continue
        grounding = _locate(str(pred.value), doc, min_support, mode)
        out.append(pred if grounding is None else pred.model_copy(update={"grounding": grounding}))
    return out


def ambiguity_penalty(score: float, n_matches: int, mode: str) -> float:
    """Discount a match ``score`` by the ambiguity of its location (``n_matches``
    equally-good places). ``uniform`` = ``score/m`` is the calibrated estimate:
    under a uniform prior over the ``m`` equally-good matches, exactly one being
    the true source, P(a given match is the source) = ``1/m``, so ``score/m``
    estimates P(correct provenance). ``sqrt``/``log`` are softer monotone
    alternatives; ``none`` leaves the score unpenalized.
    """
    if n_matches <= 1 or mode == "none":
        return score
    if mode == "sqrt":
        return score / (n_matches**0.5)
    if mode == "log":
        return score / (1.0 + math.log(n_matches))
    if mode == "uniform":
        return score / n_matches
    raise ValueError(f"unknown penalty_mode {mode!r} (none|uniform|sqrt|log)")


def _locate(
    value: str, doc: Document, min_support: float, penalty_mode: str = "uniform"
) -> Grounding | None:
    target = _normalize_for_match(value)
    if not target:
        return None
    best: Grounding | None = None
    best_support = min_support
    for page in doc.pages:
        found = _best_window(target, page)
        if found is None:
            continue
        bbox, score, n_matches = found
        # Ambiguity penalty: a value that matches equally well at n distinct
        # places (short/common tokens like a bare "2") is not reliably located,
        # so its provenance is uncertain. Discounting support turns a coincidental
        # match into low confidence -> review, instead of a falsely-confident
        # grounding. (Grounding-sweep P2; the CORD failure mode.)
        support = ambiguity_penalty(score, n_matches, penalty_mode)
        if support >= best_support:
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


def _best_window(
    target: str, page: Page
) -> tuple[tuple[float, float, float, float], float, int] | None:
    """Best contiguous word window by string similarity, plus a match count.

    Returns ``(bbox, score, n_matches)`` where ``n_matches`` is how many
    distinct page locations match at (near-)best score — the ambiguity of the
    location, which ``_locate`` uses to discount support.

    Fast path: correctly extracted values match some window verbatim, so an
    exact normalized comparison runs first. The fuzzy scan only runs for values
    with no verbatim match, and prunes windows whose length caps similarity.
    """
    words = page.words
    if not words:
        return None
    n_target_tokens = max(1, len(target.split()))
    max_width = n_target_tokens + 1
    normalized = [_normalize_for_match(w.text) for w in words]

    exact_bbox: tuple[float, float, float, float] | None = None
    exact_count = 0
    for start in range(len(words)):
        joined = normalized[start]
        for width in range(1, min(max_width, len(words) - start) + 1):
            if width > 1:
                joined = f"{joined} {normalized[start + width - 1]}"
            if joined == target:
                if exact_bbox is None:
                    exact_bbox = _window_bbox(words[start : start + width])
                exact_count += 1
                break  # count each start position once
    if exact_bbox is not None:
        return exact_bbox, 1.0, exact_count

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

    eps = 1e-9
    best_score = 0.0
    best_bbox: tuple[float, float, float, float] | None = None
    n_near = 0
    for start in starts:
        joined = normalized[start]
        for width in range(1, min(max_width, len(words) - start) + 1):
            if width > 1:
                joined = f"{joined} {normalized[start + width - 1]}"
            # similarity <= 1 - |len difference| / max(len): prune hopeless windows
            len_cap = 1.0 - abs(len(joined) - len(target)) / max(len(joined), len(target), 1)
            if len_cap < best_score - eps:
                continue
            score = normalized_levenshtein_similarity(joined, target)
            if score > best_score + eps:
                best_score = score
                best_bbox = _window_bbox(words[start : start + width])
                n_near = 1
            elif score > 0 and abs(score - best_score) <= eps:
                n_near += 1
    if best_bbox is None:
        return None
    return best_bbox, best_score, max(1, n_near)


_LONG_TARGET_TOKENS = 12


def _best_window_by_tokens(
    target_tokens: list[str], words: list, normalized: list[str]
) -> tuple[tuple[float, float, float, float], float, int] | None:
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
    return best_bbox, best_score, 1  # paragraph-length values are effectively unique


def _char_span(target: str, page: Page) -> tuple[int, int] | None:
    if not page.text:
        return None
    idx = page.text.casefold().find(target)
    if idx == -1:
        return None
    return (idx, idx + len(target))
