"""dots.ocr adapter (local, single-GPU second extractor).

Requires the dots.ocr weights served via transformers; see
https://github.com/rednote-hilab/dots.ocr. SDK use is isolated here; parsed
layout tokens go through the shared ``_ocr_common`` pathway.
"""

from __future__ import annotations

import json

from verifydoc.adapters._ocr_common import OCRToken, predictions_from_ocr_tokens
from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Schema

_PROMPT = "Please output the layout information from the image, including text and bbox."


class DotsOCRAdapter(ExtractorAdapter):
    name = "dots-ocr"

    def __init__(self, model_id: str = "rednote-hilab/dots.ocr") -> None:  # pragma: no cover
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "DotsOCRAdapter requires torch + transformers: " "pip install torch transformers"
            ) from exc
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto", device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:  # pragma: no cover
        tokens: list[OCRToken] = []
        for page in doc.pages:
            if page.image_path is None:
                continue
            raw = self._generate(page.image_path)
            for item in json.loads(raw):
                x0, y0, x1, y1 = item["bbox"]
                tokens.append(
                    OCRToken(
                        text=item.get("text", ""),
                        bbox=(
                            x0 / page.width,
                            y0 / page.height,
                            x1 / page.width,
                            y1 / page.height,
                        ),
                        score=float(item.get("score", 0.9)),
                        page=page.page,
                    )
                )
        return predictions_from_ocr_tokens(doc.doc_id, tokens, schema)

    def _generate(self, image_path: str) -> str:  # pragma: no cover
        from PIL import Image

        image = Image.open(image_path)
        inputs = self._processor(text=_PROMPT, images=image, return_tensors="pt").to(
            self._model.device
        )
        output = self._model.generate(**inputs, max_new_tokens=4096)
        return str(self._processor.batch_decode(output, skip_special_tokens=True)[0])
