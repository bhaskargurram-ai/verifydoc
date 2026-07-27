# VerifyDocBench Leaderboard

How extractors rank on **per-field trust**, not just accuracy. Every row is scored
by the same decoupled harness (`scripts/run_benchmark.py`) on the released genuine
captures; numbers link to the reproducer. To appear here: open a PR adding your
adapter (`verifydoc/adapters/`) + prediction dump, and the harness re-scores it
(see "Submit" below). Baselines are reproducible end-to-end:
`scripts/campaign2026/` (seed-pinned, 40 doc-level splits, nominal α).

## Per-field trust on genuine CORD receipts (α = 0.10, held nominal, doc-level splits)

| Extractor | fields (docs) | correct | grounding gap | verb. AUROC | coverage @10% (risk) | tier |
|---|---|---|---|---|---|---|
| claude-sonnet-5 | 13,859 (800) | .490 | +.352 [.33,.37] | .845 [.83,.86] | **0.326 (0.097)** | practical (split-protocol + HGB fusion) |
| claude-sonnet-5 | 13,859 (800) | .490 | — | — | **0.171 (0.068)** | rigorous field-iid PAC (LTT × support-bin) |
| claude-sonnet-5 | 13,859 (800) | .490 | — | — | 0.140 (0.051) | cluster-corrected (n_eff) |
| claude-sonnet-5 | 13,859 (800) | .490 | — | — | 0.060 (0.020) | rigorous doc-iid PAC |
| claude-haiku-4-5 | 5,341 (400) | .567 | +.151 [.11,.20] | .610 [.58,.64] | 0.167 (0.093) | practical (frozen-config confirmation) |
| Qwen2.5-14B (open, vLLM) | 6,168 (398) | .534 | +.276 [.24,.32] | .684 [.66,.70] | 0.149 (0.099) | practical (frozen-config confirmation) |
| RapidOCR (floor) | CORD val | low-recall | +.83 | ~.50 | ~0 (forces review) | floor baseline |
| PaddleOCR PP-OCRv5 (floor) | CORD val | low-recall | +.84 | ~.50 | ~0 (forces review) | floor baseline |

"Tier" names the guarantee level (see the method paper's validity ladder); violation
fractions and split dispersion are in `paper/generated/real-runs/campaign-2026-07-26/`.

## Multilingual + forms (grounding gap / verb. AUROC, doc-clustered CIs)

| slice | fields (docs) | correct | gap | AUROC |
|---|---|---|---|---|
| FUNSD (forms, merged) | 1,999 (175) | .676 | +.240 [.16,.33] | .835 |
| XFUND-fr | 772 (44) | .742 | +.592 [.49,.68] | .905 |
| XFUND-es | 632 (43) | .734 | +.643 [.50,.76] | .838 |
| XFUND-zh | 461 (27) | .586 | +.300 [.18,.42] | .778 |
| XFUND-de | 523 (42) | .771 | +.229 [+.03,.48] | .796 |

## Human-gold audit (3 annotators, Fleiss' κ=0.83)

Automatic labels err **one-sidedly pessimistic** (0% false-optimism on CORD);
the production accepted set's human-verified selective risk is **1.3% [0.2%, 4.8%]**
against its 10% budget (`paper/generated/real-runs/campaign-2026-07-26/human_gold_report.json`).

## Submit a system

1. Implement `ExtractorAdapter` in `verifydoc/adapters/<yours>.py` (the isolation
   boundary — no model SDK imports outside `adapters/`).
2. Produce a rich per-field dump: `scripts/apivlm_perfield_rich.py --dataset cord
   --provider <p> --model <m> --out data/<yours>.json` (or your own capture with
   the same record schema: value, verbalized, consistency, grounded, support,
   entailment, correct, doc_id).
3. Run the harness: `python scripts/campaign2026/run_thresholds.py --out results.json`
   and `run_fusion_score.py` — both are seed-pinned and regression-gated.
4. Open a PR with the adapter + dump path + the result JSON; maintainers re-run
   the harness before merging the row.

**Held-out protocol.** The public captures above are the **dev** splits (all labels
public, for method development). A held-out **test** split (200 CORD docs + 100
FUNSD docs, gold withheld from the repo, SHA-256 of the gold file published in
`benchmark/SPLITS.md`) is scored by maintainers on request for leaderboard rows
claiming a new best — this keeps the headline numbers honest as the community
iterates. Dev/test doc-id lists are versioned in `benchmark/SPLITS.md`.

## What counts as a win here

Not accuracy. A system wins by (a) higher coverage at a held nominal risk budget
with the guarantee form stated, (b) a better-calibrated per-field confidence
(ECE/Brier on the released dumps), or (c) a stronger model-agnostic trust signal
(higher error-ranking AUROC). Hallucination and omission are scored separately,
always.
