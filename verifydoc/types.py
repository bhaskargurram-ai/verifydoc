"""Core typed contracts shared by every VerifyDoc stage.

Every stage (ingest -> adapter -> confidence -> calibration -> grounding ->
policy -> report) communicates exclusively through these types, and the eval
harness scores anything that emits ``FieldPrediction``.

Path convention: leaf fields are addressed by dotted paths with explicit array
indices, e.g. ``items[2].unit_price``. Schema leaves use ``[]`` (any index),
e.g. ``items[].unit_price``; :func:`schema_path` maps a concrete path onto its
schema leaf.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

ScoringRule = Literal["exact", "numeric", "semantic"]
Decision = Literal["accept", "review"]

_INDEX_RE = re.compile(r"\[\d+\]")


class Grounding(BaseModel):
    """Provenance for a value: where on the page it was read from.

    ``bbox`` is ``(x0, y0, x1, y1)`` in normalized page coordinates in
    ``[0, 1]``; ``char_span`` is a ``[start, end)`` span into the page's text
    layer; ``support`` in ``[0, 1]`` scores how well the source region
    supports the value (e.g. IoU with a retrieval hit or string-match score).
    """

    page: int = Field(ge=0)
    bbox: tuple[float, float, float, float] | None = None
    char_span: tuple[int, int] | None = None
    support: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_geometry(self) -> Grounding:
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if x1 < x0 or y1 < y0:
                raise ValueError(f"bbox must satisfy x1>=x0, y1>=y0, got {self.bbox}")
        if self.char_span is not None and self.char_span[1] < self.char_span[0]:
            raise ValueError(f"char_span must satisfy end>=start, got {self.char_span}")
        return self


class FieldPrediction(BaseModel):
    """One extracted leaf field with its reliability contract.

    ``meta`` carries adapter-provided raw signals consumed by the confidence
    stage (e.g. ``token_logprobs: list[float]``, ``verbalized_confidence:
    float``) without coupling stages to any model SDK.
    """

    path: str
    value: Any = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    grounding: Grounding | None = None
    decision: Decision = "review"
    meta: dict[str, Any] = Field(default_factory=dict)


class FieldGold(BaseModel):
    """Gold value for one leaf field, with its scoring rule and source box."""

    path: str
    value: Any = None
    scoring: ScoringRule = "exact"
    numeric_tol: float = Field(default=1e-6, ge=0.0)
    gold_box: Grounding | None = None


class Word(BaseModel):
    """One text-layer token with its normalized bbox."""

    text: str
    bbox: tuple[float, float, float, float]


class Page(BaseModel):
    """One document page: geometry plus optional text layer."""

    page: int = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    text: str | None = None
    words: list[Word] = Field(default_factory=list)
    image_path: str | None = None


class Document(BaseModel):
    """A document to extract from (already ingested)."""

    doc_id: str
    source_path: str | None = None
    pages: list[Page] = Field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


class SchemaLeaf(BaseModel):
    """One leaf of the target schema, annotated with its scoring rule.

    This is the executable-schema pattern (ExtractBench): each field declares
    how it is scored, so the evaluator is data-driven, not hard-coded.
    """

    path: str
    type: Literal["string", "number", "integer", "boolean"] = "string"
    scoring: ScoringRule = "exact"
    numeric_tol: float = Field(default=1e-6, ge=0.0)
    required: bool = True


class Schema(BaseModel):
    """Target extraction schema: a set of scored leaves plus the raw source."""

    name: str = "schema"
    leaves: list[SchemaLeaf] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    def leaf(self, concrete_path: str) -> SchemaLeaf | None:
        """Return the schema leaf matching a concrete path (indices ignored)."""
        wanted = schema_path(concrete_path)
        for lf in self.leaves:
            if lf.path == wanted:
                return lf
        return None

    @classmethod
    def from_json_schema(cls, raw: dict[str, Any], name: str = "schema") -> Schema:
        """Build a scored-leaf schema from a JSON Schema dict.

        Leaves may carry ``x-scoring`` (``exact`` | ``numeric`` | ``semantic``)
        and ``x-numeric-tol``; defaults are ``exact`` for strings/booleans and
        ``numeric`` for numbers/integers.
        """
        leaves: list[SchemaLeaf] = []
        _walk_json_schema(raw, "", leaves, required=True)
        return cls(name=name, leaves=leaves, raw=raw)


def _walk_json_schema(
    node: dict[str, Any], prefix: str, out: list[SchemaLeaf], required: bool
) -> None:
    node_type = node.get("type", "object")
    if node_type == "object":
        req = set(node.get("required", []))
        for key, sub in node.get("properties", {}).items():
            sub_prefix = f"{prefix}.{key}" if prefix else key
            _walk_json_schema(sub, sub_prefix, out, required=key in req)
    elif node_type == "array":
        items = node.get("items", {"type": "string"})
        _walk_json_schema(items, f"{prefix}[]", out, required=required)
    else:
        leaf_type = (
            node_type if node_type in ("string", "number", "integer", "boolean") else "string"
        )
        default_scoring: ScoringRule = "numeric" if leaf_type in ("number", "integer") else "exact"
        scoring: ScoringRule = node.get("x-scoring", default_scoring)
        out.append(
            SchemaLeaf(
                path=prefix,
                type=leaf_type,  # type: ignore[arg-type]
                scoring=scoring,
                numeric_tol=float(node.get("x-numeric-tol", 1e-6)),
                required=required,
            )
        )


def schema_path(concrete_path: str) -> str:
    """Map a concrete leaf path onto its schema-leaf path (``items[2]`` -> ``items[]``)."""
    return _INDEX_RE.sub("[]", concrete_path)


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON into ``{leaf_path: value}`` using the path convention."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            sub = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(val, sub))
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            out.update(flatten_json(val, f"{prefix}[{i}]"))
    else:
        if prefix:
            out[prefix] = obj
    return out


def unflatten_json(flat: dict[str, Any]) -> Any:
    """Inverse of :func:`flatten_json` (missing array slots become ``None``)."""
    root: dict[str, Any] = {}
    for path, value in flat.items():
        _insert_path(root, path, value)
    return _listify(root)


_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _insert_path(root: dict[str, Any], path: str, value: Any) -> None:
    tokens: list[str | int] = []
    for match in _TOKEN_RE.finditer(path):
        key, idx = match.groups()
        tokens.append(int(idx) if idx is not None else key)
    node: Any = root
    for tok, nxt in zip(tokens, tokens[1:] + [None]):
        if nxt is None:
            node[tok] = value
        else:
            node = node.setdefault(tok, {})


def _listify(node: Any) -> Any:
    """Convert int-keyed dicts (array placeholders) back into lists."""
    if not isinstance(node, dict):
        return node
    if node and all(isinstance(k, int) for k in node):
        size = max(node) + 1
        return [_listify(node.get(i)) for i in range(size)]
    return {k: _listify(v) for k, v in node.items()}
