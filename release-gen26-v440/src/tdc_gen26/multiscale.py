from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class ScaleDecision:
    scale: str
    reason: str
    probe: bool
    directional_review: bool


def adaptive_separability_gate(
    decision_flip_probability: float,
    corrective_harmful_separability: float,
    local_uncertainty: float,
    meso_uncertainty: float,
    global_uncertainty: float,
    long_horizon_risk: float,
) -> ScaleDecision:
    """MESO default + selective zoom + separability-aware directional review."""
    pflip = max(0.0, min(1.0, decision_flip_probability))
    sep = max(0.0, min(1.0, corrective_harmful_separability))
    uncertainties = {
        "LOCAL8": local_uncertainty,
        "MESO18": meso_uncertainty,
        "GLOBAL128": global_uncertainty,
    }

    scale = "MESO18"
    reason = "stable_default"
    if pflip >= 0.60 and local_uncertainty + 0.05 < meso_uncertainty:
        scale = "LOCAL8"
        reason = "decision_fragility_zoom_in"
    elif long_horizon_risk >= 0.65 or global_uncertainty + 0.05 < meso_uncertainty:
        scale = "GLOBAL128"
        reason = "long_horizon_zoom_out"

    probe = pflip >= 0.55
    directional_review = probe and sep >= 0.58
    return ScaleDecision(scale=scale, reason=reason, probe=probe, directional_review=directional_review)


def scale_diagnostics(decision: ScaleDecision) -> Dict[str, object]:
    return {
        "scale": decision.scale,
        "scale_reason": decision.reason,
        "probe": decision.probe,
        "directional_review": decision.directional_review,
    }
