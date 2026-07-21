"""VerifyDoc review UI: page render + green/yellow/red fields + click-through.

Run: streamlit run ui/streamlit_app.py  (pip install 'verifydoc[ui]')

The reviewer sees exactly what the abstention layer decided: green fields were
auto-accepted at the operating point, red fields need eyes — and clicking a
field highlights the source region the value was read from.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from verifydoc.adapters import get_adapter
from verifydoc.pipeline import verify

st.set_page_config(page_title="VerifyDoc Review", layout="wide")
st.title("VerifyDoc — trust layer for document → JSON extraction")

with st.sidebar:
    st.header("Run")
    doc_file = st.file_uploader("Document (.txt / .pdf)", type=["txt", "pdf"])
    schema_file = st.file_uploader("JSON Schema", type=["json"])
    adapter_name = st.selectbox("Adapter", ["text-search", "docling", "api-vlm"])
    k = st.slider("Self-consistency samples (k)", 1, 9, 3)
    threshold = st.slider("Accept threshold", 0.0, 1.0, 0.8, 0.05)
    run = st.button("Verify", type="primary", use_container_width=True)

if run and doc_file is not None and schema_file is not None:
    with tempfile.TemporaryDirectory() as tmp:
        doc_path = Path(tmp) / doc_file.name
        doc_path.write_bytes(doc_file.getvalue())
        schema_path = Path(tmp) / "schema.json"
        schema_path.write_bytes(schema_file.getvalue())
        result = verify(
            doc_path,
            schema_path,
            adapter=get_adapter(adapter_name),
            k=k,
            threshold=threshold,
        )
    st.session_state["result"] = result
    st.session_state["doc_text"] = doc_file.getvalue().decode("utf-8", errors="replace")

if "result" in st.session_state:
    result = st.session_state["result"]
    left, right = st.columns([1, 1])

    with left:
        st.subheader(
            f"{result.doc_id}: {result.n_accepted} auto-accepted, " f"{result.n_review} for review"
        )
        selected = st.radio(
            "Fields",
            options=range(len(result.fields)),
            format_func=lambda i: (
                f"{'🟢' if result.fields[i].decision == 'accept' else '🔴'} "
                f"{result.fields[i].path} = {result.fields[i].value!r} "
                f"({result.fields[i].confidence:.2f})"
            ),
        )
        field = result.fields[selected]
        st.json(
            {
                "path": field.path,
                "value": field.value,
                "confidence": round(field.confidence, 4),
                "decision": field.decision,
                "grounding": field.grounding.model_dump() if field.grounding else None,
            }
        )
        st.download_button(
            "Download verified JSON",
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            file_name=f"{result.doc_id}.verified.json",
            mime="application/json",
        )

    with right:
        st.subheader("Source")
        text = st.session_state.get("doc_text", "")
        grounding = result.fields[selected].grounding
        if grounding and grounding.char_span and text:
            lo, hi = grounding.char_span
            st.markdown(
                f"```\n{text[:lo]}\n```\n"
                f"**:orange[{text[lo:hi]}]**  ← source span "
                f"(support {grounding.support:.2f})\n"
                f"```\n{text[hi:]}\n```"
            )
        elif text:
            st.code(text)
            if result.fields[selected].value is not None:
                st.warning("No grounding found — this value may be hallucinated.")
else:
    st.info("Upload a document and schema, then press **Verify**.")
    st.markdown(
        "Every extracted field returns `value + confidence + grounding + "
        "accept/review` — a human reviews only the fields that are actually "
        "uncertain, with click-through to the source."
    )
