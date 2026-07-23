"""Agentic layer: trust-gated extraction loops built on the verify pipeline.

The abstention policy says *which fields to trust*; these agents act on it —
repairing low-confidence fields through escalating tiers and routing only the
residue to a human. No model SDK is imported here; a tier is just an adapter.
"""

from verifydoc.agents.ensemble import adjudicate, ensemble_verify
from verifydoc.agents.repair import (
    AgenticResult,
    RepairAttempt,
    RepairTier,
    agentic_verify,
    merge_repairs,
)
from verifydoc.agents.schema_infer import (
    HeuristicSchemaProposer,
    SchemaProposer,
    infer_schema,
    verify_auto,
)

__all__ = [
    "AgenticResult",
    "HeuristicSchemaProposer",
    "RepairAttempt",
    "RepairTier",
    "SchemaProposer",
    "adjudicate",
    "agentic_verify",
    "ensemble_verify",
    "infer_schema",
    "merge_repairs",
    "verify_auto",
]
