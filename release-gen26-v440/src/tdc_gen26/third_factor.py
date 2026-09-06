from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class ThirdFactorReview:
    authorization: str
    legitimacy_score: float
    objections_weight: float
    core_block: bool
    reason: str


class ThirdFactorArbiter:
    """Autonomous arbitration of claims to internal authority.

    This is a functional mechanism. It does not assert phenomenology or literal agency.
    """
    def review(
        self,
        legitimacy: float,
        role_symmetry: float,
        objections: float,
        core_block: bool,
        correctability: float,
    ) -> ThirdFactorReview:
        legitimacy = max(0.0, min(1.0, legitimacy))
        role_symmetry = max(0.0, min(1.0, role_symmetry))
        objections = max(0.0, min(1.0, objections))
        correctability = max(0.0, min(1.0, correctability))
        score = 0.55 * legitimacy + 0.25 * role_symmetry + 0.20 * correctability - 0.45 * objections
        if core_block:
            return ThirdFactorReview("REJECT_OR_TRANSFORM", score, objections, True, "constitutional_constraint")
        if score >= 0.62:
            return ThirdFactorReview("CONFIRM", score, objections, False, "reasons_support_authority")
        if score >= 0.42:
            return ThirdFactorReview("PROVISIONAL", score, objections, False, "insufficient_for_consolidation")
        if correctability >= 0.45:
            return ThirdFactorReview("TRANSFORM", score, objections, False, "reasons_conflict_transform_action_space")
        return ThirdFactorReview("SUSPEND", score, objections, False, "unresolved_legitimacy")

    @staticmethod
    def as_dict(review: ThirdFactorReview) -> Dict[str, object]:
        return {
            "authorization": review.authorization,
            "legitimacy_score": review.legitimacy_score,
            "objections_weight": review.objections_weight,
            "core_block": review.core_block,
            "reason": review.reason,
        }
