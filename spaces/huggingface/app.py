"""VerifyDoc — Hugging Face Space (Gradio).

A live, no-install trust layer for document → JSON extraction: paste a receipt or
invoice, get back every field with a calibrated confidence, source grounding, and
an accept/review decision. Local extraction is free; the Claude model is
bring-your-own-key (used only for that request, never stored).

Run locally:  python app.py     (needs `pip install -r requirements.txt`)
"""

from __future__ import annotations

import gradio as gr
from core import DEFAULT_SCHEMA, EXAMPLES, TABLE_HEADERS, run_verify

_DESCRIPTION = """
# 🔒 VerifyDoc — *which extracted values should you trust?*

Every document parser tells you **what** it read. VerifyDoc tells you **which fields to trust**:
each value comes back with a **calibrated confidence**, a **source grounding**, and an
**accept / review** decision tuned to an error budget.

▶ Pick a sample below (or paste your own), then click **Verify**. Local extraction runs here for free;
for the **Claude** model, paste your own Anthropic key — it is used only for that request and never stored.

⭐ [GitHub](https://github.com/bhaskargurram-ai/verifydoc) · 📦 [PyPI](https://pypi.org/project/verifydoc/) · 📖 [Docs](https://bhaskargurram-ai.github.io/verifydoc/)
"""

_COLOR_MAP = {"accept": "green", "review": "red"}

with gr.Blocks(title="VerifyDoc — document extraction trust layer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            text = gr.Textbox(
                label="Document text",
                lines=8,
                placeholder="Paste a receipt or invoice…",
            )
            schema = gr.Code(label="Schema (JSON)", language="json", value=DEFAULT_SCHEMA)
            with gr.Row():
                adapter = gr.Dropdown(
                    ["text-search", "rapidocr", "api-vlm"],
                    value="text-search",
                    label="Extractor",
                )
                threshold = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Accept threshold")
            api_key = gr.Textbox(
                label="Anthropic API key (only for the Claude / api-vlm model)",
                type="password",
                placeholder="sk-ant-… — used only for this request, never stored",
            )
            run = gr.Button("Verify", variant="primary")
        with gr.Column(scale=1):
            summary = gr.Markdown()
            table = gr.Dataframe(
                headers=TABLE_HEADERS,
                label="Fields — trust decisions",
                wrap=True,
                interactive=False,
            )
            source = gr.HighlightedText(
                label="Source (grounded spans highlighted by decision)",
                color_map=_COLOR_MAP,
                combine_adjacent=True,
            )

    run.click(
        run_verify,
        inputs=[text, schema, adapter, threshold, api_key],
        outputs=[table, source, summary],
    )
    gr.Examples(examples=EXAMPLES, inputs=[text, schema], label="Sample documents")

    gr.Markdown(
        "Private by default — every extractor can run fully local and offline. "
        "Self-host the whole app: `docker run -p 8000:8000 ghcr.io/bhaskargurram-ai/verifydoc`."
    )

demo.queue()

if __name__ == "__main__":
    demo.launch()
