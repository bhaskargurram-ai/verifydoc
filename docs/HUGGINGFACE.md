# Hugging Face Space (live demo)

VerifyDoc ships two Space builds. **Static is free**; Gradio/Docker Spaces
require a Hugging Face PRO subscription, so use the static Space for a free,
public demo.

- **Space:** <https://huggingface.co/spaces/bhaskar1225/verifydoc>

## Recommended: free **static** Space

[`spaces/huggingface-static/`](https://github.com/bhaskargurram-ai/verifydoc/tree/main/spaces/huggingface-static)
is the VerifyDoc review cockpit served as a static page that calls the hosted
VerifyDoc API — no server-side runtime, so it hosts **free** on Hugging Face.
`index.html` is generated from the packaged cockpit
(`verifydoc/server/review.html`) by `scripts/build_hf_static.py`; regenerate it
whenever the cockpit changes:

```bash
python scripts/build_hf_static.py           # writes spaces/huggingface-static/index.html
```

### Publish / update the Space (needs your HF token)

```bash
pip install -U huggingface_hub
hf auth login                               # write token from hf.co/settings/tokens

# create the Space explicitly as STATIC (free) — this avoids the CLI defaulting
# to a Gradio Space, which needs PRO (HTTP 402):
hf repo create bhaskar1225/verifydoc --repo-type space --space-sdk static -y

# then upload the static Space contents (index.html + README.md):
hf upload bhaskar1225/verifydoc spaces/huggingface-static . \
  --repo-type space --commit-message "Deploy VerifyDoc demo"
```

If your `hf` CLI version doesn't accept `--space-sdk`, use the Python API, which
always does:

```python
from huggingface_hub import create_repo, upload_folder
create_repo("bhaskar1225/verifydoc", repo_type="space", space_sdk="static", exist_ok=True)
upload_folder(repo_id="bhaskar1225/verifydoc", repo_type="space",
              folder_path="spaces/huggingface-static")
```

The Space rebuilds on every upload; `README.md` (`sdk: static`) keeps it free.

> The static page calls the hosted API cross-origin, which the backend allows
> via CORS (`VERIFYDOC_CORS_ORIGINS`, default `*` for the public demo). Point it
> at your own backend by re-running the build with `--api-base https://your-host`.

## Alternative: Gradio Space (needs HF PRO, or run locally)

[`spaces/huggingface/`](https://github.com/bhaskargurram-ai/verifydoc/tree/main/spaces/huggingface)
is a self-contained Gradio app that runs `verifydoc.verify` in-process (no
external backend). Hosting it on Hugging Face needs PRO, but it runs anywhere:

```bash
cd spaces/huggingface
pip install -r requirements.txt
python app.py          # http://localhost:7860
```

Its `core.py` keeps the verify logic free of any Gradio import, so it is
unit-tested offline in CI (`tests/test_hf_space.py`).
