# Campaign 2026-07-26 harnesses (Paper A "validity ladder" + Paper B measurement study)

Self-contained research harnesses behind the 2026-07 papers. All use 40 fixed
document-level calibration/test splits (seed 7), report at NOMINAL alpha (no
tolerance bands), and carry bit-exact sanity/regression gates chained to the
original experiment runs.

- `run_covariate_ablation.py --data-dir data` — Mondrian conditioning-covariate
  ablation (provenance vs non-provenance taxonomies) under the add-one rule.
- `run_thresholds.py` — the threshold-procedure grid: add-one (pooled/Mondrian),
  document-level add-one, Learn-then-Test with exact binomial tails
  (Holm / fixed-sequence / mixed; per-group; cluster-corrected n_eff; doc-level
  Hoeffding–Bentkus), boundary-snapped candidate grids. Produces the money table.
- `run_fusion_score.py` — the score-side grid: baseline LR fusion vs
  HistGradientBoosting fusion under the leakage-safe fit/val SPLIT PROTOCOL,
  with engineered features and entailment/fieldtype ablations.

Inputs: rich per-field dumps `data/apivlm_perfield_rich_*.json` produced by
`scripts/apivlm_perfield_rich.py` (see Paper B for capture details). The sanity
gates check against the archived pre-campaign dumps
(`archive/pre-campaign-2026-07-26/data/`, local-only, not committed).
Result JSONs used by the papers: `paper/generated/real-runs/campaign-2026-07-26/`
(local-only). Full experiment log: `paper/reviews/_overnight_campaign_2026-07-26.md`
(local-only).
