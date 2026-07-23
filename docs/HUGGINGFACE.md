# Hugging Face Space (live Gradio demo)

VerifyDoc ships a ready-to-deploy Gradio demo in
[`spaces/huggingface/`](https://github.com/bhaskargurram-ai/verifydoc/tree/main/spaces/huggingface)
(`app.py`, `core.py`, `requirements.txt`, `README.md` with the Space frontmatter).
It reuses `verifydoc.verify` — paste a document + schema and every field comes
back with confidence, grounding, and an accept/review decision. Local extraction
runs in the Space for free; the Claude model is bring-your-own-key.

## Try it

- **Space:** <https://huggingface.co/spaces/bhaskargurram-ai/verifydoc>

## Run locally

```bash
cd spaces/huggingface
pip install -r requirements.txt
python app.py          # http://localhost:7860
```

## Publish / update the Space (needs your HF token)

Pushing to the Hub requires an interactive Hugging Face login, so run this
yourself (the demo files are already in the repo):

```bash
pip install -U huggingface_hub
huggingface-cli login                     # paste a write token from hf.co/settings/tokens

# one-time create (Gradio SDK):
huggingface-cli repo create verifydoc --type space --space_sdk gradio

# upload the Space contents (app.py, core.py, requirements.txt, README.md):
huggingface-cli upload bhaskargurram-ai/verifydoc spaces/huggingface . \
  --repo-type space --commit-message "Deploy VerifyDoc demo"
```

The Space rebuilds automatically on each upload. The pinned `requirements.txt`
installs `verifydoc[pdf,ocr,api]>=0.10.0` from PyPI, so publish the matching
PyPI release first (tag `v0.10.0`).

> The Space's `core.py` keeps the verify logic free of any Gradio import, so it
> is unit-tested offline in CI (`tests/test_hf_space.py`) even though Gradio and
> the hosted runtime are not installed there.
