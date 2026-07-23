"""XFUND loader (multilingual form KIE, 7 languages) -> VerifyDocBench.

XFUND extends FUNSD to Chinese, Japanese, Spanish, French, Italian, German,
and Portuguese. Each language ships a ``train``/``val`` split of annotated
forms with the same key->answer linking structure as FUNSD, so we reuse
FUNSD's :func:`bench_from_annotation` to build ``BenchDocument`` items.

The raw release (github.com/doc-analysis/XFUND) packages per-language JSON
+ image zips; we stream from the HuggingFace hub
(``naver-clova-ix/xfund`` or compatible) and JSON-cache under ``data/``.
Unit tests use fixture annotations only (golden rule #5).

License note (benchmark/card.md): XFUND is released for research use; we
ship the loader + our added splits, never the images.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.datasets.funsd import bench_from_annotation
from benchmark.datasets.synthetic import BenchDocument

XFUND_LANGS = ("de", "es", "fr", "it", "ja", "pt", "zh")

XFUND_URL_TMPL = (
    "https://github.com/doc-analysis/XFUND/releases/download/v1.0/"
    "{lang}.{split}.json"
)


def load(
    lang: str = "en",
    split: str = "val",
    limit: int | None = None,
    cache_dir: str | Path = "data",
) -> list[BenchDocument]:  # pragma: no cover - network on first call
    """Load XFUND for one language (downloads JSON once into ``cache_dir``).

    ``lang`` is an ISO 639-1 code from :data:`XFUND_LANGS`; ``split`` is
    ``"train"`` or ``"val"``. FUNSD itself is the English subset and is
    loaded via :mod:`benchmark.datasets.funsd`.
    """
    import urllib.request

    if lang not in XFUND_LANGS:
        raise ValueError(
            f"XFUND lang {lang!r} not available; choose from {XFUND_LANGS}"
        )
    if split not in ("train", "val"):
        raise ValueError(f"XFUND split must be 'train' or 'val', got {split!r}")

    root = Path(cache_dir) / "xfund"
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{lang}.{split}.json"
    if not json_path.exists():
        urllib.request.urlretrieve(XFUND_URL_TMPL.format(lang=lang, split=split), json_path)

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    documents = raw.get("documents", [])
    out: list[BenchDocument] = []
    for i, doc in enumerate(documents):
        if limit is not None and i >= limit:
            break
        form = doc.get("form", doc.get("document", []))
        img = doc.get("img", {})
        width = int(img.get("width", 1000))
        height = int(img.get("height", 1000))
        # XFUND doc ids are strings (e.g. "de_val_0"); index numerically for a stable id
        doc_id = f"xfund-{lang}-{split}-{i:04d}"
        item = bench_from_annotation(doc_id, {"form": form}, width, height)
        if item is not None:
            out.append(item)
    return out
