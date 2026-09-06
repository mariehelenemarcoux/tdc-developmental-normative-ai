from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ZoomPlan:
    depth: int
    direction: str
    cost: float
    reason: str


class RecursiveZoomController:
    """Self-similar zoom operator with explicit compute-cost discipline."""
    def __init__(self, max_depth: int = 4, unit_cost: float = 0.08) -> None:
        self.max_depth=max_depth
        self.unit_cost=unit_cost

    def plan(
        self,
        decision_fragility: float,
        separability: float,
        long_horizon_risk: float,
        compute_budget: float,
    ) -> ZoomPlan:
        f=max(0.0,min(1.0,decision_fragility))
        s=max(0.0,min(1.0,separability))
        lr=max(0.0,min(1.0,long_horizon_risk))
        budget=max(0.0,float(compute_budget))
        if f<0.45 and lr<0.60:
            return ZoomPlan(0,"NONE",0.0,"stable_meso_default")
        direction="OUT" if lr>=0.65 and f<0.75 else "IN"
        # Depth is earned by fragility and separability, never by fragility alone.
        raw=int(round(self.max_depth * f * (0.35+0.65*s)))
        affordable=int(budget//self.unit_cost) if self.unit_cost>0 else self.max_depth
        depth=max(0,min(self.max_depth,raw,affordable))
        if depth==0:
            return ZoomPlan(0,"NONE",0.0,"insufficient_budget_or_separability")
        return ZoomPlan(depth,direction,depth*self.unit_cost,"decision_relevant_recursive_zoom")
