# Running real extractors (GPU runbook)

The same harness, with a real extractor instead of the simulated one
(issue #3). Reproduces `paper/generated/{cord,funsd}-{rapidocr,paddleocr}/`.

## Hard-won environment notes (read first)

These were discovered running on RunPod; they save hours:

- **PaddleOCR needs a GPU architecture paddle supports.** PaddlePaddle's
  precompiled wheels have **no kernels for Blackwell (sm_120, e.g. RTX PRO
  6000 / 5090)** — you get `RuntimeError: Unsupported GPU architecture`. Use
  Ampere (A100 sm_80, A5000/A6000 sm_86) or Ada (RTX 4090 / L4 / L40 sm_89).
- **Pin `paddlex==3.1.0`** to match `paddleocr==3.1.0`, or the constructor
  dies with `PaddlePredictorOption.__init__() takes 1 positional argument`.
- **Pin `langchain<0.2`** (paddlex imports the removed `langchain.docstore`).
- **RapidOCR is architecture-independent** (ONNX Runtime) — it runs anywhere,
  CPU or GPU, and is the most robust real-OCR option. Prefer it when the GPU
  is exotic.
- **dots.ocr** loads with `transformers==4.51.3` + `qwen_vl_utils`, but raw
  `transformers.generate` fights its custom model; its intended runtime is
  **vllm serving** — wire the adapter to a vllm endpoint (issue #3).

## Setup

```bash
git clone https://github.com/bhaskargurram-ai/verifydoc && cd verifydoc
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,data]"

# Option A — RapidOCR (recommended, architecture-independent)
pip install rapidocr onnxruntime          # or onnxruntime-gpu for speed

# Option B — PaddleOCR (needs a paddle-supported GPU)
pip install "paddlepaddle-gpu==3.1.0" -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install "paddleocr==3.1.0" "paddlex==3.1.0" "langchain<0.2" "langchain-community<0.2"
```

## Smoke test one receipt

```bash
python - <<'EOF'
from benchmark.datasets import cord
from verifydoc.adapters import get_adapter

item = cord.load(split="validation", limit=1, with_images=True)[0]
adapter = get_adapter("rapidocr")          # or "paddleocr-vl"
for p in adapter.extract(item.doc, item.schema):
    print(p.path, p.value, p.meta.get("token_logprobs"))
EOF
```

## Full run

```bash
python scripts/run_benchmark.py --config configs/cord-rapidocr.yaml   --out paper/generated/cord-rapidocr
python scripts/run_benchmark.py --config configs/funsd-rapidocr.yaml  --out paper/generated/funsd-rapidocr
python scripts/run_benchmark.py --config configs/cord-paddleocr.yaml  --out paper/generated/cord-paddleocr   # paddle GPU
python scripts/run_benchmark.py --config configs/funsd-paddleocr.yaml --out paper/generated/funsd-paddleocr
```

The harness dispatches `extractor:` through the adapter registry
(`mock` | `text-search` | `rapidocr` | `paddleocr-vl` | `dots-ocr` |
`docling` | `api-vlm`) and `dataset:` (`synthetic` | `cord` | `funsd`).
CORD/FUNSD images are exported automatically (`with_images`).

## What to report

Pin model versions in the table header (`rapidocr==x`, `paddleocr==3.1.0`,
GPU + driver). Results feed the paper via `make results`; see
`paper/generated/REAL_MODELS_RESULTS.md` for the current numbers and reading.

## Cloud

Any Ampere/Ada single-GPU pod (RTX 4090 ≈ $0.4–0.7/hr) runs everything except
Blackwell-only cards for paddle. The full sweep is an hours-not-days job.
```
