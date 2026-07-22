"""Extraction-quality metrics (PROJECT.md §5.A).

Implements: field P/R/F1, exact-match accuracy, CER/WER, ANLS, JSON per-field
scoring with separate omission and hallucination rates (executable-schema
pattern, after ExtractBench), TEDS / TEDS-Struct (Zhong et al., tree edit
distance over HTML table trees), and GriTS Top/Con/Loc (Smock et al.,
arXiv:2203.12555, factored most-similar-substructure alignment).

The eval harness is model-agnostic: it scores any ``list[FieldPrediction]``
against ``list[FieldGold]``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Literal

from verifydoc.types import FieldGold, FieldPrediction

# ---------------------------------------------------------------------------
# String-distance primitives
# ---------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert/delete/substitute, unit costs)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    """``1 - dist / max(len)``; two empty strings are perfectly similar."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def cer(pred: str, gold: str) -> float:
    """Character Error Rate = edit distance / len(gold). Can exceed 1.

    Empty gold: 0.0 for an empty prediction, else edits over a denominator
    of 1 (so a spurious prediction against empty gold is heavily penalized).
    """
    if not gold:
        return 0.0 if not pred else float(len(pred))
    return levenshtein(pred, gold) / len(gold)


def wer(pred: str, gold: str) -> float:
    """Word Error Rate: token-level edit distance / gold token count."""
    pred_tokens, gold_tokens = pred.split(), gold.split()
    if not gold_tokens:
        return 0.0 if not pred_tokens else float(len(pred_tokens))
    return _token_edit_distance(pred_tokens, gold_tokens) / len(gold_tokens)


def _token_edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        cur = [i]
        for j, tb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = cur
    return prev[-1]


def anls(preds: list[str], golds: list[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (DocVQA convention).

    Per pair: NLS = 1 - normalized distance; scores below ``tau`` are zeroed.
    """
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length")
    if not preds:
        return 0.0
    total = 0.0
    for p, g in zip(preds, golds):
        sim = normalized_levenshtein_similarity(p, g)
        total += sim if sim >= tau else 0.0
    return total / len(preds)


# ---------------------------------------------------------------------------
# Value normalization + per-rule correctness (schema-as-spec)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_CURRENCY_CODE_RE = re.compile(r"^[A-Za-z]{2,4}\s+|\s+[A-Za-z]{2,4}$")
_NUM_CLEAN_RE = re.compile(r"[$€£%\s]")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def normalize_text(value: Any) -> str:
    """Canonical string form: str, strip, collapse whitespace."""
    return _WS_RE.sub(" ", str(value)).strip()


def parse_number(value: Any) -> float | None:
    """Parse a numeric value out of common document formats ($1,234.50 etc.).

    Comma handling: 3-digit groups (``1,234.50``) are thousands separators;
    otherwise a comma with no dot (``42,5``) is a decimal comma.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    # Strip leading/trailing alphabetic currency codes (RM, Rp, USD, etc.)
    # before numeric cleaning so they don't end up glued to digits.
    text = _CURRENCY_CODE_RE.sub("", str(value)).strip()
    text = _NUM_CLEAN_RE.sub("", text)
    if _THOUSANDS_RE.match(text) or ("," in text and "." in text):
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def value_correct(pred_value: Any, gold: FieldGold, semantic_tau: float = 0.5) -> bool:
    """Score one predicted value under its gold field's scoring rule.

    - ``exact``: casefolded, whitespace-normalized string equality.
    - ``numeric``: parsed numbers within ``gold.numeric_tol`` (absolute).
    - ``semantic``: normalized Levenshtein similarity >= ``semantic_tau``.
      # DECISION: string-similarity proxy for semantic equivalence keeps unit
      # tests offline and deterministic; an embedding/LLM judge can be swapped
      # in behind this same rule name for the paper's semantic fields.
    """
    if pred_value is None and gold.value is None:
        return True
    if pred_value is None or gold.value is None:
        return False
    if gold.scoring == "numeric":
        p, g = parse_number(pred_value), parse_number(gold.value)
        if p is None or g is None:
            return False
        return abs(p - g) <= gold.numeric_tol
    p_text = normalize_text(pred_value).casefold()
    g_text = normalize_text(gold.value).casefold()
    if gold.scoring == "exact":
        return p_text == g_text
    return normalized_levenshtein_similarity(p_text, g_text) >= semantic_tau


def exact_match(pred_value: Any, gold_value: Any) -> bool:
    """Exact-match accuracy predicate: normalized string equality."""
    if pred_value is None or gold_value is None:
        return pred_value is None and gold_value is None
    return normalize_text(pred_value).casefold() == normalize_text(gold_value).casefold()


# ---------------------------------------------------------------------------
# JSON per-field scoring: P/R/F1 + omission vs hallucination
# ---------------------------------------------------------------------------


@dataclass
class FieldScore:
    """Per-predicted-field outcome; feeds calibration/selective metrics."""

    path: str
    correct: bool
    confidence: float
    predicted_value: Any
    gold_value: Any
    status: Literal["matched", "hallucinated"]


@dataclass
class ExtractionReport:
    """Field-level scoring of one or more documents' predictions."""

    n_gold: int
    n_predicted: int
    n_correct: int
    precision: float
    recall: float
    f1: float
    exact_match_rate: float
    omission_rate: float
    hallucination_rate: float
    field_scores: list[FieldScore] = field(default_factory=list)
    omitted_paths: list[str] = field(default_factory=list)
    hallucinated_paths: list[str] = field(default_factory=list)


def score_fields(
    predictions: list[FieldPrediction],
    golds: list[FieldGold],
    semantic_tau: float = 0.5,
) -> ExtractionReport:
    """Score predictions against gold under each field's own rule.

    # DECISION (omission vs hallucination, after ExtractBench):
    #  - omission      = a gold field with no prediction (or predicted None);
    #    omission_rate = omitted / n_gold.
    #  - hallucination = a predicted field whose path has no gold;
    #    hallucination_rate = hallucinated / n_predicted.
    # They are disjoint, reported separately, and hallucinated fields count
    # against precision. ``field_scores`` covers every *predicted* field
    # (matched + hallucinated) with its confidence, so the selective layer is
    # scored on exactly what the extractor asserted.
    """
    gold_by_path = {g.path: g for g in golds}
    if len(gold_by_path) != len(golds):
        raise ValueError("duplicate gold paths")

    scores: list[FieldScore] = []
    hallucinated: list[str] = []
    predicted_paths: set[str] = set()
    n_correct = 0
    n_exact = 0

    for pred in predictions:
        if pred.value is None:
            continue  # an explicit None is an omission, not an assertion
        predicted_paths.add(pred.path)
        gold = gold_by_path.get(pred.path)
        if gold is None:
            hallucinated.append(pred.path)
            scores.append(
                FieldScore(
                    path=pred.path,
                    correct=False,
                    confidence=pred.confidence,
                    predicted_value=pred.value,
                    gold_value=None,
                    status="hallucinated",
                )
            )
            continue
        correct = value_correct(pred.value, gold, semantic_tau=semantic_tau)
        n_correct += correct
        n_exact += exact_match(pred.value, gold.value)
        scores.append(
            FieldScore(
                path=pred.path,
                correct=correct,
                confidence=pred.confidence,
                predicted_value=pred.value,
                gold_value=gold.value,
                status="matched",
            )
        )

    omitted = [g.path for g in golds if g.path not in predicted_paths]
    n_gold = len(golds)
    n_predicted = len(scores)
    precision = n_correct / n_predicted if n_predicted else 0.0
    recall = n_correct / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return ExtractionReport(
        n_gold=n_gold,
        n_predicted=n_predicted,
        n_correct=n_correct,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match_rate=n_exact / n_gold if n_gold else 0.0,
        omission_rate=len(omitted) / n_gold if n_gold else 0.0,
        hallucination_rate=len(hallucinated) / n_predicted if n_predicted else 0.0,
        field_scores=scores,
        omitted_paths=omitted,
        hallucinated_paths=hallucinated,
    )


# ---------------------------------------------------------------------------
# TEDS / TEDS-Struct (Zhong et al.): tree edit distance over HTML tables
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    text: str = ""
    children: list[_Node] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


_TABLE_TAGS = {"table", "thead", "tbody", "tr", "td", "th"}


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.root: _Node | None = None
        self._stack: list[_Node] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _TABLE_TAGS:
            return
        attr = dict(attrs)
        node = _Node(
            tag="td" if tag == "th" else tag,
            colspan=int(attr.get("colspan") or 1),
            rowspan=int(attr.get("rowspan") or 1),
        )
        if self._stack:
            self._stack[-1].children.append(node)
        elif self.root is None:
            self.root = node
        self._stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        if tag in _TABLE_TAGS and self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._stack and self._stack[-1].tag == "td":
            self._stack[-1].text += data


def _parse_table(html: str) -> _Node:
    parser = _TableHTMLParser()
    parser.feed(html)
    if parser.root is None:
        raise ValueError("no <table> found in HTML")
    return parser.root


def _postorder(root: _Node) -> tuple[list[_Node], list[int]]:
    """Postorder nodes + leftmost-leaf-descendant index per node (Zhang-Shasha)."""
    nodes: list[_Node] = []
    lmld: list[int] = []

    def walk(node: _Node) -> int:
        first_leaf = -1
        for child in node.children:
            leaf = walk(child)
            if first_leaf == -1:
                first_leaf = leaf
        nodes.append(node)
        idx = len(nodes) - 1
        lmld.append(first_leaf if first_leaf != -1 else idx)
        return lmld[idx]

    walk(root)
    return nodes, lmld


def _keyroots(lmld: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    for i, leaf in enumerate(lmld):
        seen[leaf] = i  # last (highest) node for each leftmost leaf
    return sorted(seen.values())


def _rename_cost(a: _Node, b: _Node, struct_only: bool) -> float:
    if a.tag != b.tag:
        return 1.0
    if a.tag == "td":
        if a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        if struct_only:
            return 0.0
        return 1.0 - normalized_levenshtein_similarity(a.text.strip(), b.text.strip())
    return 0.0


def _tree_edit_distance(t1: _Node, t2: _Node, struct_only: bool) -> float:
    """Zhang-Shasha ordered tree edit distance with TEDS costs (unit ins/del)."""
    n1, l1 = _postorder(t1)
    n2, l2 = _postorder(t2)
    kr1, kr2 = _keyroots(l1), _keyroots(l2)
    dist = [[0.0] * len(n2) for _ in n1]

    for i in kr1:
        for j in kr2:
            _forest_dist(i, j, n1, n2, l1, l2, dist, struct_only)
    return dist[len(n1) - 1][len(n2) - 1]


def _forest_dist(
    i: int,
    j: int,
    n1: list[_Node],
    n2: list[_Node],
    l1: list[int],
    l2: list[int],
    dist: list[list[float]],
    struct_only: bool,
) -> None:
    li, lj = l1[i], l2[j]
    rows, cols = i - li + 2, j - lj + 2
    fd = [[0.0] * cols for _ in range(rows)]
    for x in range(1, rows):
        fd[x][0] = fd[x - 1][0] + 1.0
    for y in range(1, cols):
        fd[0][y] = fd[0][y - 1] + 1.0
    for x in range(1, rows):
        for y in range(1, cols):
            di, dj = li + x - 1, lj + y - 1
            if l1[di] == li and l2[dj] == lj:
                fd[x][y] = min(
                    fd[x - 1][y] + 1.0,
                    fd[x][y - 1] + 1.0,
                    fd[x - 1][y - 1] + _rename_cost(n1[di], n2[dj], struct_only),
                )
                dist[di][dj] = fd[x][y]
            else:
                fd[x][y] = min(
                    fd[x - 1][y] + 1.0,
                    fd[x][y - 1] + 1.0,
                    fd[l1[di] - li][l2[dj] - lj] + dist[di][dj],
                )


def teds(pred_html: str, gold_html: str, struct_only: bool = False) -> float:
    """TEDS = 1 - d(T_pred, T_gold) / max(|T_pred|, |T_gold|).

    ``struct_only=True`` gives TEDS-Struct (cell text ignored).
    """
    t_pred, t_gold = _parse_table(pred_html), _parse_table(gold_html)
    d = _tree_edit_distance(t_pred, t_gold, struct_only=struct_only)
    return 1.0 - d / max(t_pred.size(), t_gold.size())


# ---------------------------------------------------------------------------
# GriTS (Smock et al., arXiv:2203.12555)
# ---------------------------------------------------------------------------


@dataclass
class GridCell:
    """One entry of a table's grid matrix (spanning cells repeat their entry)."""

    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    bbox: tuple[float, float, float, float] | None = None


CellSim = Callable[[GridCell, GridCell], float]


def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _sim_content(a: GridCell, b: GridCell) -> float:
    ta, tb = a.text.strip(), b.text.strip()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return 2.0 * _lcs_len(ta, tb) / (len(ta) + len(tb))


def _sim_topology(a: GridCell, b: GridCell) -> float:
    inter = min(a.rowspan, b.rowspan) * min(a.colspan, b.colspan)
    union = a.rowspan * a.colspan + b.rowspan * b.colspan - inter
    return inter / union if union else 0.0


def _sim_location(a: GridCell, b: GridCell) -> float:
    if a.bbox is None or b.bbox is None:
        return 0.0
    from verifydoc.eval.grounding import iou

    return iou(a.bbox, b.bbox)


_GRITS_SIMS: dict[str, CellSim] = {
    "con": _sim_content,
    "top": _sim_topology,
    "loc": _sim_location,
}


def _weighted_lcs(xs: list[Any], ys: list[Any], sim: Callable[[Any, Any], float]) -> float:
    """Maximum-total-similarity monotone alignment (weighted LCS DP)."""
    prev = [0.0] * (len(ys) + 1)
    for x in xs:
        cur = [0.0]
        for j, y in enumerate(ys, start=1):
            cur.append(max(prev[j], cur[j - 1], prev[j - 1] + sim(x, y)))
        prev = cur
    return prev[-1]


def grits(
    pred: list[list[GridCell]],
    gold: list[list[GridCell]],
    variant: Literal["con", "top", "loc"] = "con",
) -> tuple[float, float, float]:
    """GriTS precision/recall/F-score for a variant (Con | Top | Loc).

    # DECISION: 2D-MSS is NP-hard; per the GriTS paper we use the factored
    # approximation — a nested weighted-LCS DP (rows aligned by DP, each
    # row pair scored by a weighted-LCS over its cells). Cell similarity:
    # Con = normalized longest-common-subsequence of text; Top = IoU of
    # (rowspan x colspan) extents; Loc = IoU of cell bboxes.
    """
    if not pred or not gold:
        return (0.0, 0.0, 0.0)
    cell_sim = _GRITS_SIMS[variant]

    def row_sim(ra: list[GridCell], rb: list[GridCell]) -> float:
        return _weighted_lcs(ra, rb, cell_sim)

    total = _weighted_lcs(pred, gold, row_sim)
    n_pred = sum(len(r) for r in pred)
    n_gold = sum(len(r) for r in gold)
    precision = total / n_pred if n_pred else 0.0
    recall = total / n_gold if n_gold else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (precision, recall, f)
