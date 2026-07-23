# Changelog

## Unreleased

Agentic app, local private extraction, and top-repo polish.

- **Local HF-model adapter** (`hf-vlm`): fully-private extraction via a local
  transformers model — no API, nothing leaves the machine. (#41)
- **`verify_batch`** + **observability** (`verifydoc/observability.py`): structured
  verification events + JSON logging + optional OpenTelemetry span export. (#26)
- **Export layer** (`verifydoc/export.py`): verified results → CSV / JSONL carrying
  the trust columns (confidence, decision, grounded, page). (#40)
- **Messaging bots** now extract with the configured model (Claude when a key is
  present, else the local baseline); **`verifydoc-bot`** runs the Telegram bot by
  long-polling with no public webhook. (#24)
- **Grouping-taxonomy ablation + characterization** in the harness: coverage per
  provenance taxonomy vs pooled conformal, the calibration-split-selected
  partition, and a predictor of when grouping helps. (#25)
- **DocILE + XFUND** (multilingual) dataset loaders. (#38, thanks @MasRama)
- **Docs website** (MkDocs Material → GitHub Pages) + **MCP-registry `server.json`**;
  **Docker → ghcr.io** publish workflow. (#37, #42)
- Shipped **`py.typed`** (PEP 561); `SECURITY.md`, `.github/FUNDING.yml`; README
  badges, comparison table, and one-command installs (uvx/pipx/docker). (#27)
- Agentic + batch **examples**; **CLI test** coverage. (#36, #39)


## v0.9.0 — 2026-07-22

Free maintenance + more datasets.

- **Free AI PR review via GitHub Models** (`pr-review-free.yml`): reviews every
  PR using the built-in `GITHUB_TOKEN` — no API key, no cost. The paid Claude
  reviewer is now an optional upgrade. `docs/AUTOMATION.md` documents the
  zero-cost maintenance stack.
- **SROIE loader** (`benchmark/datasets/sroie.py`): receipts with company/date/
  address/total gold fields + located boxes; wired into the harness dispatch.
- Contributor `good first issue`s opened for DocILE/XFUND, observability,
  adaptive-k budgeting, entailment-verified grounding, and currency parsing.


## v0.8.0 — 2026-07-22

Adoption, automation, and repo hygiene.

- **Framework integrations** (`verifydoc.integrations`): `verify_instructor_result`
  for Instructor/Pydantic/Outlines/Marvin and `VerifiedExtractor` for LangChain —
  add the trust layer in a few lines; no framework dependency imported.
- **GitHub automation**: Dependabot (pip + actions), CodeQL security scanning,
  PR labeler, stale triage, a secret-gated Claude code-review dev-agent, and a
  packaging build job + coverage artifact in CI.
- **Repo hygiene**: removed dead code; benchmark output now goes to a
  git-ignored `results/` dir.
- **Proprietary paper removed from GitHub**: the `paper/` directory (LaTeX
  write-up + generated tables/figures) is purged from the tree and all history
  and kept local-only; the open-source library, benchmark harness, and
  reproducibility scripts remain public.


## v0.7.0 — 2026-07-22

Literature-grounded hardening (4-agent SOTA sweep) + production DX.

- **Ambiguity-penalized grounding** — discount support by the number of
  equally-good matches, so coincidental short-value matches no longer ground
  falsely. This made the novel method work on real data: CORD coverage
  0.01→0.72 at a held 10% risk (previously no lift, guarantee violated); FUNSD
  0.24→0.84 at 2%.
- **Dynamic schemas** — `Schema.to_json_schema`/`json_schema` reconstruct a
  JSON Schema from leaves (any dynamic/FUNSD schema works with the API-VLM);
  `Schema.from_pydantic` + `verify_model` for Pydantic-native extraction.
- **SOTA metrics** — AUGRC (NeurIPS 2024) and kernel-smoothed ECE (ICLR 2024)
  added to the harness.
- **Related work** — `docs/RELATED_WORK.md` + a fully-cited paper Related Work
  positioning VerifyDoc vs Beyond Logprobs, Cleanlab TLM, CRC-certify,
  conformal factuality, Gibbs–Candès; paper now 8pp, compiles.


## v0.6.0 — 2026-07-22

Journal-grade rigor: larger N, real-data method, and labeling reliability.

- **Grounder numeric-aware matching** — strips thousands separators + currency,
  so `45500` grounds to `45,500`/`$45,500`. Gold grounding on real CORD went
  from partial to ~100%; this unlocked the real-data method study.
- **Novel method on REAL data at scale** (`scripts/grounded_conformal_real.py`):
  CORD train (~5.5k fields) + full FUNSD (~2.6k fields), 40-split marginal
  evaluation. On FUNSD, grounding-conditioned conformal lifts coverage from
  0.24 to 0.71 at a 2% risk guarantee; honest boundary characterization on CORD
  (short numeric values → ambiguous grounding → limited gain).
- **Inter-annotator agreement** — `cohens_kappa`/`fleiss_kappa` in
  `eval/stats.py`; a labeling module (`verifydoc/labeling.py`) + `verifydoc iaa`
  CLI; a two-protocol agreement study (CORD κ=0.78 robust; FUNSD κ=0.10 shows
  free-text labels are protocol-dependent). Labeling guide + tooling shipped.
- **Paper** grown to 7pp with real-data method section, labeling-reliability
  section, and a strengthened, honest limitations section; refs tidied.

## v0.5.0 — 2026-07-22

A novel method, a frontier-VLM comparison, and an agent-facing application.

- **Novel method — grounding-conditioned (Mondrian) conformal risk control**
  (`verifydoc/calibration/grouped_conformal.py`): conditions the abstention
  threshold on provenance, preserving a finite-sample per-group risk guarantee
  while accepting well-grounded fields at a lower bar. Controlled study
  (`scripts/grouped_conformal_experiment.py`): **+0.50 mean coverage** at a
  fixed 5% risk in the uninformative-confidence regime where pooled conformal
  accepts almost nothing. First use of provenance as the conditioning taxonomy
  for conformal in document extraction.
- **Frontier VLM results** (`claude-sonnet-5`, k=3, on CORD): recall 0.56 but a
  **0.48 hallucination rate**; verbalized confidence is informative (AUROC
  0.86) for the VLM yet useless (0.50) for OCR pipelines — a cross-extractor
  finding the single-model literature misses. Cross-extractor summary in
  `paper/generated/REAL_MODELS_RESULTS.md`.
- **MCP server** (`verifydoc-mcp`): exposes `verify_extraction` over the Model
  Context Protocol so AI agents extract documents with confidence + grounding +
  accept/review — a drop-in trust layer for the agentic era (`docs/MCP.md`).
- Paper elevated to a method contribution (6pp, compiles; novel result in the
  abstract and a dedicated section). `mcp` install extra.

## v0.4.0 — 2026-07-21

Paper-ready: fair API-VLM comparison + a compilable paper.

- **Vendor-neutral API-VLM extractor** (`verifydoc/adapters/api_vlm.py`):
  OpenAI + Anthropic clients behind one `CompletionClient` protocol, with
  **temperature-based k-sample sampling** so self-consistency consensus and
  verbalized confidence are non-degenerate (the fair comparison the paper
  needs). Harness `adapter_kwargs` passthrough; `configs/cord-apivlm.yaml`;
  `ocr` and `api` install extras.
- **Compilable paper** (`paper/main.tex`) with real RapidOCR/PaddleOCR/
  synthetic numbers, auto-generated LaTeX result tables
  (`scripts/tables_to_latex.py`, booktabs) and `refs.bib`; `make paper`.
- **Docs**: `docs/how-it-works.md`; README documentation index; runnable
  `examples/`.
- Repo hygiene: removed internal build-tooling notes; tidied PROJECT.md.

## v0.3.0 — 2026-07-21

First **real-model** results (issue #3).

- **RapidOCR adapter** (PP-OCRv6 via ONNX Runtime) — architecture-independent,
  the robust real-OCR default; runs where paddle's GPU wheels don't.
- **PaddleOCR adapter** validated end-to-end (PP-OCRv5 pipeline) on a real GPU.
- **Real results on CORD + FUNSD** for both extractors, committed under
  `paper/generated/` with an honest reading (`REAL_MODELS_RESULTS.md`):
  grounding/learned signals rank errors (AUROC 0.74–0.89) while
  verbalized/consensus don't; grounding-conditioned gap +0.83/+0.84. The
  floor field-finder's low recall forces conformal abstention at 2–5% — the
  documented motivation for the selective framing.
- **Harness**: `extractor:` dispatch via the adapter registry; CORD/FUNSD
  image export (`with_images`); schema `x-aliases` for on-page label variants.
- **Learned combiner** (`verifydoc/confidence/learned.py`) as a harness row.
- **Grounder ~1000× faster** (exact-match fast path, token-anchored fuzzy
  scan, token-overlap for paragraph values) — FUNSD grounding hours → seconds.
- **GPU runbook** (`docs/REAL_MODELS.md`) with the hard-won version pins
  (`paddlex==3.1.0`, `langchain<0.2`, Blackwell caveat, dots.ocr→vllm).
- dots.ocr: adapter stub validated to load/generate with `transformers==4.51.3`;
  full run deferred to its vllm serving path (issue #3).

## v0.2.0 — 2026-07-21

Real data, faster grounding, learned fusion.

- **CORD v2 slice**: real receipts with real text layers (word quads) and
  located gold boxes; Indonesian thousands-comma prices score numerically.
- **FUNSD slice**: 199 scanned forms; question→answer links become gold
  fields with exact annotated answer boxes.
- **Harness**: `dataset:` (synthetic | cord | funsd) and `extractor:` (any
  adapter registry name) dispatch; `make results` regenerates all slices.
- **Learned combiner**: logistic fusion over signal values + missing
  indicators, fit on the calibration split; reported as a `learned` row in
  every table (ablation vs the transparent weighted mean).
- **Grounder performance (~1000×)**: exact-match fast path, token-anchored
  fuzzy scan with length pruning, token-overlap scoring for paragraph-length
  values. FUNSD run: hours → 8 seconds.
- **Fix**: text-search label matching uses word boundaries ("total" no longer
  fires inside "Subtotal") — caught by the harness itself.
- Demo GIF rendered from a real pipeline run; GPU runbook
  (docs/REAL_MODELS.md) + pinned real-model config; paper skeleton.

## v0.1.0 — 2026-07-21

First public release.

- **Core types**: `FieldPrediction` / `FieldGold` / `Grounding` / `Document` /
  executable `Schema` (per-leaf scoring rules), path flatten/unflatten.
- **Adapters**: pluggable `ExtractorAdapter` interface; mock (canned + seeded
  noisy), text-search baseline, PaddleOCR-VL, dots.ocr, Docling/MinerU output,
  API-VLM (injectable client).
- **Confidence signals**: token-prob, verbalized, k-sample consensus voting,
  grounding-based, weighted combined.
- **Calibrators**: temperature, Platt, isotonic, histogram + split-conformal
  abstention with finite-sample E[risk] ≤ α guarantee; split-disjointness
  guards (never tune on test, enforced in code).
- **Grounding stage**: value → page/bbox/char-span attachment with support.
- **Policy**: empirical + conformal target-risk thresholds; accept/review.
- **Eval harness (VerifyDocBench scorer)**: extraction (P/R/F1, exact,
  CER/WER, ANLS, TEDS/TEDS-Struct, GriTS Con/Top/Loc, omission vs
  hallucination), calibration (ECE 15-bin, Adaptive ECE, MCE, Brier, NLL,
  TCE, reliability), selective (RC, AURC, E-AURC, Coverage@Risk,
  Risk@Coverage, Acc@k, AUROC, AUPR, FPR@95%TPR), grounding (IoU acc @
  0.5/0.7, mean IoU, span F1, grounding-conditioned correctness), stats
  (bootstrap CIs, paired permutation/bootstrap tests). All numerically
  regression-tested against hand-computed fixtures.
- **Benchmark**: deterministic synthetic slice with gold values *and* gold
  boxes; CORD v2 loader; `make results` regenerates all tables/figures from
  pinned configs.
- **UX**: typer CLI (`verifydoc extract`), Streamlit review UI with
  click-through grounding.
