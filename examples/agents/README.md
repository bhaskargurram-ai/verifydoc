# Agentic examples

VerifyDoc is the **trust primitive** for document agents: instead of acting on
whatever an extractor returns, the agent acts only on fields that clear a
calibrated confidence bar and are grounded to the source, and escalates the rest.

- **`agent_review_loop.py`** — a framework-free demonstration of the trust-gated
  loop: `extract → grade (VerifyDoc) → accept / escalate / re-extract`. The
  escalate branch is where a LangGraph `interrupt()`, an OpenAI-Agents
  `needs_approval` tool, or a VerifyDoc review queue plugs in.

```bash
python examples/agents/agent_review_loop.py
```

**Why this matters:** an agent that reads a document and acts on a silently-wrong
value fails silently. VerifyDoc turns "the model said so" into an auditable
accept/review decision with a source citation — the missing safety gate for
document-reading agents.

See issue [#29](https://github.com/bhaskargurram-ai/verifydoc/issues/29) for the
full LangGraph/CrewAI integration (contributions welcome).
