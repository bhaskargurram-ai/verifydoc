#!/usr/bin/env python
"""Render docs/demo.gif from a REAL pipeline run.

A canned adapter returns one silently-wrong value (total: $1,234.50 ->
$1,432.50). The grounder cannot locate it on the page, its confidence
collapses, and the policy routes it to review — the GIF shows exactly what
the library computed, drawn with PIL. Usage: python scripts/make_demo_gif.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifydoc.adapters.mock import MockAdapter  # noqa: E402
from verifydoc.ingest import document_from_text  # noqa: E402
from verifydoc.pipeline import verify  # noqa: E402
from verifydoc.types import FieldPrediction, Schema  # noqa: E402

W, H = 960, 640
PAGE_W = 560
FONT_PATH = "/System/Library/Fonts/Menlo.ttc"

INVOICE_TEXT = """ACME CORPORATION
Invoice ID: INV-2024-001
Vendor: ACME Corporation
Date: 2024-01-15
Subtotal: $1,141.00
Tax: $93.50
Total: $1,234.50
Thank you for your business"""

SCHEMA = Schema.from_json_schema(
    {
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "vendor": {"type": "string", "x-scoring": "semantic"},
            "date": {"type": "string"},
            "total": {"type": "number", "x-numeric-tol": 0.01},
        },
    },
    name="invoice",
)

CANNED = [
    FieldPrediction(path="invoice_id", value="INV-2024-001"),
    FieldPrediction(path="vendor", value="ACME Corporation"),
    FieldPrediction(path="date", value="2024-01-15"),
    FieldPrediction(path="total", value="$1,432.50"),  # silently wrong: digits swapped
]


def fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    try:
        return (
            ImageFont.truetype(FONT_PATH, 17),
            ImageFont.truetype(FONT_PATH, 15),
            ImageFont.truetype(FONT_PATH, 22),
        )
    except OSError:  # non-mac fallback
        base = ImageFont.load_default()
        return base, base, base


def page_canvas(doc) -> Image.Image:
    """Draw the document words at their (normalized) layout positions."""
    img = Image.new("RGB", (W, H), "#f5f2ec")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 20 + PAGE_W, H - 20], fill="white", outline="#c8c2b6", width=2)
    body, _small, _big = fonts()
    for word in doc.pages[0].words:
        x0, y0, _x1, _y1 = word.bbox
        draw.text(
            (30 + x0 * PAGE_W * 1.55, 40 + y0 * (H - 100) * 0.9), word.text, fill="#222", font=body
        )
    return img


def word_box_px(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        28 + x0 * PAGE_W * 1.55,
        37 + y0 * (H - 100) * 0.9,
        36 + x1 * PAGE_W * 1.55,
        41 + y1 * (H - 100) * 0.9 + 8,
    )


def panel(draw: ImageDraw.ImageDraw, result, upto: int, headline: str) -> None:
    body, small, big = fonts()
    x = 620
    draw.text((x, 40), "VerifyDoc", fill="#111", font=big)
    draw.text((x, 72), headline, fill="#555", font=small)
    y = 120
    for i, f in enumerate(result.fields):
        if i >= upto:
            break
        ok = f.decision == "accept"
        color = "#1a7f37" if ok else "#c62828"
        mark = "ACCEPT" if ok else "REVIEW"
        draw.text((x, y), f"{f.path}", fill="#333", font=small)
        draw.text((x, y + 18), f"= {f.value}", fill="#111", font=body)
        draw.text((x, y + 40), f"conf {f.confidence:.2f}  {mark}", fill=color, font=small)
        if not ok:
            draw.text((x, y + 58), "disagrees with the page -> review", fill="#c62828", font=small)
        y += 88
    return


def main() -> None:
    doc = document_from_text("invoice", [INVOICE_TEXT])
    adapter = MockAdapter(canned={"invoice": CANNED})
    result = verify(doc, SCHEMA, adapter=adapter, threshold=0.8)
    by_path = {f.path: f for f in result.fields}
    assert by_path["total"].decision == "review", "demo premise broke"
    assert by_path["invoice_id"].decision == "accept"

    frames: list[Image.Image] = []

    base = page_canvas(doc)
    d = ImageDraw.Draw(base)
    _body, small, big = fonts()
    d.text((620, 40), "VerifyDoc", fill="#111", font=big)
    d.text((620, 72), "document + schema in ...", fill="#555", font=small)
    frames.append(base)

    for upto in range(1, len(result.fields) + 1):
        frame = page_canvas(doc)
        d = ImageDraw.Draw(frame)
        for f in result.fields[:upto]:
            if f.grounding and f.grounding.bbox:
                color = "#1a7f37" if f.decision == "accept" else "#c62828"
                d.rectangle(word_box_px(f.grounding.bbox), outline=color, width=3)
        panel(d, result, upto, "per-field trust: conf + grounding")
        frames.append(frame)

    total = by_path["total"]
    support = total.grounding.support if total.grounding else 0.0
    final = frames[-1].copy()
    d = ImageDraw.Draw(final)
    d.rectangle([620, 476, 946, 604], fill="#fdf3f3", outline="#c62828", width=2)
    d.text((632, 488), "extracted $1,432.50 - but the page", fill="#c62828", font=small)
    d.text((632, 508), f"says $1,234.50 (support {support:.2f}).", fill="#c62828", font=small)
    d.text((632, 532), "Red box = where to look.", fill="#c62828", font=small)
    d.text((632, 562), "Human checks 1 field, not 4.", fill="#111", font=small)
    frames.extend([final, final])

    out = Path("docs/demo.gif")
    out.parent.mkdir(exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=[900, 700, 700, 700, 900, 2600, 2600],
        loop=0,
        optimize=True,
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
