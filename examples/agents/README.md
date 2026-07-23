# Agentic examples

VerifyDoc is the **trust primitive** for document agents: instead of acting on
whatever an extractor returns, the agent acts only on fields that clear a
calibrated confidence bar and are grounded to the source, and escalates the rest.

- **`agent_review_loop.py`** — a framework-free demonstration of the trust-gated
  loop: `extract → grade (VerifyDoc) → accept / escalate / re-extract`. The
  escalate branch is where a LangGraph `interrupt()`, an OpenAI-Agents
  `needs_approval` tool, or a VerifyDoc review queue plugs in.
- **`langgraph_review.py`** — the same loop as an actual LangGraph `StateGraph`:
  `extract_and_verify → (conditional edge) → human_review → finalize`, where the
  human step is a real `interrupt()`. The trust nodes are plain functions, so it
  runs (and is tested) with **zero dependencies**; install `langgraph` to get the
  interrupt-driven graph via `build_langgraph_app()`.

```bash
python examples/agents/agent_review_loop.py
python examples/agents/langgraph_review.py   # framework-free; uses LangGraph if installed
```

**Why this matters:** an agent that reads a document and acts on a silently-wrong
value fails silently. VerifyDoc turns "the model said so" into an auditable
accept/review decision with a source citation — the missing safety gate for
document-reading agents.

- **`auto_repair.py`** — the built-in agentic layer, `verifydoc.agents.agentic_verify`:
  extract → verify → **repair** `review` fields through lazy escalating tiers (a
  stronger adapter / more samples) → **escalate** the residue to a human. Tiers
  run cheapest-first and only while review fields remain, so cost scales with
  document difficulty (`n_extract_calls` makes it measurable).

```bash
python examples/agents/auto_repair.py
```

- **`ensemble_adjudication.py`** — `verifydoc.agents.ensemble_verify`: run several
  different extractors (OCR / VLM / API) and **adjudicate per field**. Where they
  agree, confidence rises; where they disagree, the best-grounded reading wins and
  the dissent is recorded — genuine splits stay `review`.

```bash
python examples/agents/ensemble_adjudication.py
```

CrewAI and OpenAI-Agents variants of the same trust gate are welcome — the
routing policy in `langgraph_review.py` is framework-agnostic.
