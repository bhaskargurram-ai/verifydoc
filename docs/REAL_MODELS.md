# Running real extractors (GPU runbook)

The shipped adapters run PaddleOCR-VL and dots.ocr on a single 24 GB GPU.
This is the paper's headline experiment (issue #3): the same harness, with a
real extractor instead of the simulated one.

## One-time setup (Linux + NVIDIA, 24 GB)

```bash
git clone https://github.com/bhaskargurram-ai/verifydoc && cd verifydoc
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pdf,data]"

# PaddleOCR-VL
pip install paddlepaddle-gpu paddleocr

# dots.ocr
pip install torch transformers accelerate
```

## Smoke test one receipt

```bash
python - <<'EOF'
from benchmark.datasets import cord
from verifydoc.adapters import get_adapter

item = cord.load(split="validation", limit=1)[0]
adapter = get_adapter("paddleocr-vl")
for p in adapter.extract(item.doc, item.schema):
    print(p.path, p.value, p.meta.get("token_logprobs"))
EOF
```

Note: OCR adapters read `page.image_path` — for CORD, export images first
(`row["image"].save(...)`) and set `image_path` on each page; a
`--with-images` flag on the CORD loader is the intended patch (part of
issue #3).

## Full run

```bash
python scripts/run_benchmark.py --config configs/cord-paddleocr.yaml \
  --out paper/generated/cord-paddleocr
```

`configs/cord-paddleocr.yaml` is checked in as the pinned experiment config;
the harness dispatches `extractor:` through the adapter registry
(`mock` | `text-search` | `paddleocr-vl` | `dots-ocr` | `docling` | `api-vlm`).

## What to report

The identical tables the simulated run produces (`calibration.md`,
`selective.md`, `conformal.md`, `grounding.md`) — swap them into the paper
via `make results`. Pin model versions in the config (`paddleocr==x.y.z`)
and record GPU + driver in the table header for reproducibility.

## Cloud options

Any of: RunPod / Lambda / Vast single RTX 4090 (24 GB) instance ≈ $0.4–0.8/hr;
the full 100-receipt sweep with k=5 sampling is an hours-not-days job.
