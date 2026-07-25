# What is genuinely novel about VerifyDoc

*A honest novelty statement for the open-source package, written from a scientist
skeptic-vs-advocate debate (2026-07). It deliberately distinguishes **scientific
novelty** (new knowledge) from **engineering / open-availability** contributions
(real, but not "novelty"). If you are evaluating whether to build on VerifyDoc, or
citing it, read the "Honest ceiling" first.*

## TL;DR — the crisp claim

> **VerifyDoc is the first open, model-agnostic software artifact that turns any
> document→JSON extractor into a per-field *trust contract* — a calibrated
> confidence, a source grounding (page/bbox/char-span), and a risk-controlled
> accept/review decision — shipped as a library, CLI, review UI, MCP server, and a
> self-enforcing evaluation harness. Its one *scientific* seed is empirical: it is
> the first to use **provenance (grounding) as the conditioning taxonomy for
> conformal risk control in document extraction** — the gap `crccertify` explicitly
> left open — and to characterize *when* that conditioning pays.**

Everything else is either an engineering/openness contribution (real and useful) or
an adopted convention we credit to prior work. We say so on purpose: over-claiming
method novelty is the fastest way to lose a reviewer's trust, and the honest story
is still a strong one.

## The five pillars, ranked and graded

| # | Pillar | What it *is* | Closest prior art | Honest grade |
|---|--------|--------------|-------------------|--------------|
| 1 | **Grounding-conditioned (Mondrian) conformal** | Mondrian/group-conditional conformal with *provenance* as the covariate | Vovk 2003 (Mondrian CP); Gibbs–Cherian–Candès 2025; CRC; `crccertify` (structured-gen abstention, left provenance open) | **Empirical first-application** — the machinery is classical; the covariate, the doc-extraction application, and the *characterization of when it helps* are the contribution. Modest, and contingent (see "Earn it"). |
| 2 | **Agentic trust-gating** (repair / adjudicate / adaptive-*k*) | Verifier-gated cascade + majority-vote ensemble + adaptive self-consistency, actuated by the *calibrated/grounded* decision | FrugalGPT cascades (Chen 2023); self-consistency (Wang 2022); adaptive-consistency (Aggarwal 2023); BSDetector | **Engineering pattern**, not method novelty. The only fresh bit is gating on the conformal decision + a require-grounded check. Ship it, don't headline it. |
| 3 | **Executable-schema trust contract** | Per-leaf scoring rule (exact/numeric/semantic) on a Pydantic/JSON-Schema leaf | **ExtractBench** (credited in our own `types.py`) | **Adopted convention.** Our only twist is bundling the scoring rule into the same object graph as confidence/grounding/decision — an API-design choice. |
| 4 | **First open MCP server / PyPI package with a trust contract** | stdio transport marshalling into `verify()`; a free package name | Box (Jan 2026), Azure DI, AWS Textract, Extend, Cleanlab TLM ship the *substance* closed | **Open-availability / engineering**, not science. Legitimately valuable (the open implementation of a monetized-closed contract) — but a priority claim, not new knowledge. |
| 5 | **VerifyDocBench: joint calibration + selective-risk + grounding** | Three standard metric families in one released harness + loaders + gold source boxes | `beyondlogprobs` (calib+selective for doc fields, no grounding); DocILE/KILE, SROIE, OCRBench-v2 (grounding); ECE/AURC lineage | **Open infrastructure + released artifact.** "First to combine three known metrics" is a harness feature; the real contribution is the *released* labeled resource (and it must earn human-IAA labels — free-text κ≈0.10 today). |

## The debate, in one paragraph each

**Advocate.** No single pillar is a theorem, but the *artifact* is novel: grounding
is load-bearing in **two** layers at once — it conditions the conformal threshold
*and* gates the agentic repair loop — producing trust behavior that neither Beyond
Logprobs nor CRC-certify deliver, and it is the first time the whole contract is
open, self-hostable, and agent-callable over MCP. Availability of a gap the field
explicitly named is itself a contribution.

**Skeptic.** By the project's own USP doc, VerifyDoc is "the open implementation" of
a contract Box/Azure/Textract/Cleanlab already ship — so the burden falls entirely
on the method/empirical deltas, and those are thin: pillar 1 is a covariate choice
for a 22-year-old technique (and the phase-2 grid shows a *non-provenance* covariate
`support×verb` ties `grounded×support`, undercutting "provenance is special");
pillars 2–4 are renamed prior patterns, self-credited adoption, and a transport
wrapper; pillar 5's calibration+selective axis was already run by Beyond Logprobs.
Honest ceiling: a strong open tool + one modest empirical finding.

## Honest ceiling (what we actually claim)

A **well-engineered open system** that (a) is the first to package a known per-field
trust contract openly (PyPI + MCP + reproducible harness), and (b) contributes one
modest, falsifiable empirical finding: **grounding is a usable per-field trust
signal, and provenance is an effective conditioning covariate for conformal
selective-risk control in document extraction — where the base score is weak and
error is heterogeneous across provenance groups (a characterization, not yet a
theorem; validated on n=3 genuine-VLM datasets).** We do *not* claim method novelty
for pillars 2–4.

## "Earn it" — what would upgrade the contingent claims (collaboration wanted)

These are open, tracked as GitHub issues — contributions welcome:

1. **Provenance-conditioning, cleanly.** One guarantee-respecting win: achieved risk
   ≤ *nominal* α under document-clustered accounting (no tolerance band), coverage
   lift > ~+0.05, with the taxonomy-selection multiplicity correction applied — **and
   an ablation showing provenance beats non-provenance covariates** (support-only,
   verbalized, field-type). If `support×verb` keeps tying `grounded×support`, the
   honest claim shrinks to "a good covariate for Mondrian CP." *(On genuine-VLM CORD
   today the control is only approximate at finite sample — see the method paper.)*
2. **Agentic trust-gating, proven.** An accuracy-at-fixed-compute experiment showing
   the trust-gated cascade / adjudicated ensemble beats a plain
   confidence-thresholded cascade and k-sample self-consistency at matched extract
   calls. Until then it is "an agent integration," not a contribution.
3. **VerifyDocBench, credible labels.** A human-labeled gold slice with reported IAA
   (free-text κ≈0.10 automatic is disqualifying for semantic fields), plus a genuinely
   new *joint* metric the three families don't give separately (grounding-conditioned
   selective risk *with a guarantee*).

## Prior art we build on (cite these, not around them)

Mondrian CP (Vovk 2003); conformal risk control / Learn-then-Test; Gibbs–Cherian–Candès
2025; `crccertify` (structured-generation abstention); Beyond Logprobs (doc-field
calibration+selective); ExtractBench (executable schema); BSDetector, FrugalGPT,
self-consistency (agentic patterns); DocILE/KILE, SROIE, OCRBench-v2 (grounding &
KIE); Box/Azure/Textract/Cleanlab TLM (the closed commercial trust contract).
