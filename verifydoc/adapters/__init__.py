"""Extractor adapters. Golden rule #1: all model-specific code lives here.

Nothing outside this package may import a model SDK; adding a new extractor
is one new file implementing ``ExtractorAdapter``.
"""

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.adapters.mock import MockAdapter
from verifydoc.adapters.text_search import TextSearchAdapter

__all__ = ["ExtractorAdapter", "MockAdapter", "TextSearchAdapter", "get_adapter"]

_REGISTRY = {
    "mock": "verifydoc.adapters.mock:MockAdapter",
    "text-search": "verifydoc.adapters.text_search:TextSearchAdapter",
    "paddleocr-vl": "verifydoc.adapters.paddleocr_vl:PaddleOCRVLAdapter",
    "dots-ocr": "verifydoc.adapters.dots_ocr:DotsOCRAdapter",
    "docling": "verifydoc.adapters.docling:DoclingAdapter",
    "api-vlm": "verifydoc.adapters.api_vlm:APIVLMAdapter",
}


def get_adapter(name: str, **kwargs: object) -> ExtractorAdapter:
    """Instantiate a registered adapter by CLI name (lazy import)."""
    import importlib
    from typing import cast

    if name not in _REGISTRY:
        raise ValueError(f"unknown adapter {name!r}; available: {sorted(_REGISTRY)}")
    module_name, _, class_name = _REGISTRY[name].partition(":")
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)
    return cast(ExtractorAdapter, adapter_cls(**kwargs))
