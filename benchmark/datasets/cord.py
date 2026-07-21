"""CORD v2 loader (receipts; permissive license) -> VerifyDocBench items.

Downloads via HuggingFace ``datasets`` (pip install verifydoc[data]); never
run in unit tests (golden rule #5). Gold values come from CORD's ground-truth
parse; ``correct_i`` is therefore computable automatically (PROJECT.md §12).
"""

from __future__ import annotations

from benchmark.datasets.synthetic import BenchDocument
from verifydoc.ingest import document_from_text
from verifydoc.types import FieldGold, Schema

CORD_SCHEMA_RAW: dict = {
    "type": "object",
    "required": ["total_price"],
    "properties": {
        "store_name": {"type": "string", "x-scoring": "semantic"},
        "total_price": {"type": "number", "x-numeric-tol": 0.01},
        "subtotal_price": {"type": "number", "x-numeric-tol": 0.01},
        "tax_price": {"type": "number", "x-numeric-tol": 0.01},
    },
}

_GT_PATHS = {
    "store_name": ("store_info", "name"),
    "total_price": ("total", "total_price"),
    "subtotal_price": ("subtotal", "subtotal_price"),
    "tax_price": ("subtotal", "tax_price"),
}


def load(
    split: str = "validation", limit: int | None = None
) -> list[BenchDocument]:  # pragma: no cover - network
    """Load CORD v2 from the HF hub (naver-clova-ix/cord-v2)."""
    import json

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("CORD loader requires: pip install 'verifydoc[data]'") from exc

    schema = Schema.from_json_schema(CORD_SCHEMA_RAW, name="cord-receipt")
    rows = load_dataset("naver-clova-ix/cord-v2", split=split)
    out: list[BenchDocument] = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        gt = json.loads(row["ground_truth"])["gt_parse"]
        golds = []
        for path, keys in _GT_PATHS.items():
            node = gt
            for key in keys:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, (str, int, float)) and str(node):
                leaf = schema.leaf(path)
                assert leaf is not None
                golds.append(
                    FieldGold(
                        path=path,
                        value=node,
                        scoring=leaf.scoring,
                        numeric_tol=leaf.numeric_tol,
                    )
                )
        if not golds:
            continue
        # text layer from the gt lines; images stay on the hub (never committed)
        lines = [
            " ".join(w["text"] for w in line["words"])
            for line in json.loads(row["ground_truth"]).get("valid_line", [])
            if isinstance(line, dict) and line.get("words")
        ]
        doc = document_from_text(f"cord-{split}-{i:05d}", ["\n".join(lines)])
        out.append(BenchDocument(doc=doc, schema=schema, golds=golds))
    return out
