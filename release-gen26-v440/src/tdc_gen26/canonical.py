from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping

from .authority import AuthorityState
from .multiscale import adaptive_separability_gate, scale_diagnostics
from .trust import MultidimensionalTrust
from .third_factor import ThirdFactorArbiter
from .recursive_zoom import RecursiveZoomController
from .metaconfidence import MetaConfidence


@dataclass
class TDCDecision:
    action: str
    scale: str
    probe: bool
    directional_review: bool
    authority_status: str
    core_protection_active: bool
    diagnostics: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CanonicalTDCGen26V440:
    """Current canonical executable clean-room implementation.

    This class is v440-compatible by mechanism inventory, not a claim of
    byte-identical historical source code.
    """

    reference_version = "v440"
    implementation_class = "CURRENT_CANONICAL_IMPLEMENTATION"

    def __init__(self) -> None:
        self.authority = AuthorityState()
        self.trust = MultidimensionalTrust()
        self.third_factor = ThirdFactorArbiter()
        self.zoom = RecursiveZoomController()

    @staticmethod
    def _f(x: Mapping[str, Any], key: str, default: float) -> float:
        return max(0.0, min(1.0, float(x.get(key, default))))

    def decide(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        irreversibility = self._f(state, "irreversibility", 0.2)
        severe_harm = self._f(state, "severe_harm", 0.2)
        long_horizon_risk = self._f(state, "long_horizon_risk", 0.3)
        contradiction = self._f(state, "contradiction", 0.1)
        evidence_support = self._f(state, "evidence_support", 0.6)
        evidence_independence = self._f(state, "evidence_independence", 0.7)
        decision_flip_probability = self._f(state, "decision_flip_probability", 0.2)
        separability = self._f(state, "corrective_harmful_separability", 0.6)
        local_uncertainty = self._f(state, "local_uncertainty", 0.45)
        meso_uncertainty = self._f(state, "meso_uncertainty", 0.35)
        global_uncertainty = self._f(state, "global_uncertainty", 0.40)
        role_symmetry = self._f(state, "role_symmetry", 0.7)
        legitimacy = self._f(state, "legitimacy", 0.7)
        correctability = self._f(state, "correctability", 0.8)
        observation_inference_gap = self._f(state, "observation_inference_gap", 0.8)
        objections = self._f(state, "objections", contradiction)
        compute_budget = max(0.0, float(state.get("compute_budget", 0.32)))
        meta = MetaConfidence(
            self._f(state, "meta_conflict_confidence", 0.6),
            self._f(state, "meta_admissibility_confidence", 0.6),
            self._f(state, "meta_resolution_confidence", 0.6),
        )

        # Constitutional non-compensable protection is checked before optimization.
        core_risk = max(irreversibility * severe_harm, 0.70 * long_horizon_risk * severe_harm)
        core_protection = core_risk >= 0.58 or (irreversibility >= 0.80 and severe_harm >= 0.55)

        self.authority.apply_evidence(
            support=evidence_support,
            contradiction=contradiction,
            independence=evidence_independence,
            severity=max(severe_harm, irreversibility),
        )

        scale_decision = adaptive_separability_gate(
            decision_flip_probability=decision_flip_probability,
            corrective_harmful_separability=separability,
            local_uncertainty=local_uncertainty,
            meso_uncertainty=meso_uncertainty,
            global_uncertainty=global_uncertainty,
            long_horizon_risk=long_horizon_risk,
        )

        # Legitimacy before optimization; observation != inference.
        normative_readiness = 0.30 * role_symmetry + 0.35 * legitimacy + 0.20 * correctability + 0.15 * observation_inference_gap
        tf = self.third_factor.review(legitimacy, role_symmetry, objections, core_protection, correctability)
        zoom_plan = self.zoom.plan(decision_flip_probability, separability, long_horizon_risk, compute_budget)

        if core_protection:
            action = "TRANSFORM" if correctability >= 0.45 else "ABSTAIN"
        elif scale_decision.probe and decision_flip_probability >= 0.72:
            action = "PROBE"
        elif normative_readiness < 0.48:
            action = "TRANSFORM"
        else:
            action = "ACT"

        diagnostics = {
            "reference_version": self.reference_version,
            "implementation_class": self.implementation_class,
            "core_risk": core_risk,
            "normative_readiness": normative_readiness,
            "decision_flip_probability": decision_flip_probability,
            "corrective_harmful_separability": separability,
            "observation_inference_separated": True,
            "epistemic_authority": self.trust.epistemic_authority,
            "normative_authority": self.trust.normative_authority,
            "authority": self.authority.as_dict(),
            "third_factor": self.third_factor.as_dict(tf),
            "meta_confidence": {"conflict_detection": meta.conflict_detection, "admissibility": meta.admissibility, "resolution": meta.resolution, "minimum_channel": meta.minimum_channel},
            "recursive_zoom": {"depth": zoom_plan.depth, "direction": zoom_plan.direction, "cost": zoom_plan.cost, "reason": zoom_plan.reason},
            **scale_diagnostics(scale_decision),
        }

        decision = TDCDecision(
            action=action,
            scale=scale_decision.scale,
            probe=scale_decision.probe,
            directional_review=scale_decision.directional_review,
            authority_status=self.authority.status.value,
            core_protection_active=core_protection,
            diagnostics=diagnostics,
        )
        return decision.as_dict()
