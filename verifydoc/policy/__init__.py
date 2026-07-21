"""Abstention policy stage: accept/review decisions at a target error rate."""

from verifydoc.policy.abstention import AbstentionPolicy, apply_policy, threshold_for_target_risk

__all__ = ["AbstentionPolicy", "apply_policy", "threshold_for_target_risk"]
