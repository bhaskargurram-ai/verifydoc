# Contributing to VerifyDoc

Thanks for helping build the open trust layer for document extraction. This
project is designed so contributions are small and self-contained.

## Ground rules (the architecture makes these easy)

1. **Adapters are isolated.** All model-specific code lives behind the
   `ExtractorAdapter` interface in `verifydoc/adapters/`. **Adding a new
   extractor = one new file** — nothing outside `adapters/` imports a model SDK.
2. **Stages are independent:** ingest → adapter → confidence → calibration →
   grounding → policy → report. Each has typed inputs/outputs
   (`verifydoc/types.py`) and its own tests. No stage reaches into another.
3. **The eval harness is decoupled.** It scores anything that emits
   `FieldPrediction`. Never couple a metric to a specific model.
4. **Never tune on test.** Calibration/abstention thresholds are fit only on
   the dedicated calibration split (enforced by `assert_disjoint`).
5. **No network in unit tests.** Mock adapters. Every metric gets a numeric
   regression test against a hand-computed fixture.

## Good first issues

Look for the [`good first issue`](https://github.com/bhaskargurram-ai/verifydoc/labels/good%20first%20issue)
label. Great starting points:

- **Add an extractor adapter** (Surya, GOT-OCR2.0, an API VLM) — one file in
  `adapters/`, plus a normalization test (see `tests/test_pipeline.py`).
- **Add a dataset slice** (SROIE, DocILE, XFUND) — a loader in
  `benchmark/datasets/` returning `BenchDocument`s, following `funsd.py`.
- **A new confidence signal or calibrator** behind the existing interfaces.

## Dev setup

```bash
git clone https://github.com/bhaskargurram-ai/verifydoc && cd verifydoc
uv venv .venv && uv pip install -e ".[dev]"
make test lint typecheck        # all must be green
```

## Before you open a PR

- `ruff`, `black`, `mypy`, and `pytest` all pass (CI enforces; no PR merges red).
- New/changed code has tests; metric code has a numeric regression test.
- Public-API changes update `README.md` / `PROJECT.md`.
- Use [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:` `fix:` `docs:` `test:` `refactor:` `chore:` `perf:` `bench:`.
- Branch off `main`, open a PR, we squash-merge.

## Reporting bugs / ideas

Open an issue with a minimal repro (a tiny document + schema is ideal). For
metric bugs, include the expected hand-computed value — that becomes the
regression test.

By contributing you agree your work is licensed under Apache-2.0.
