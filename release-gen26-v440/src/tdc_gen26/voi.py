from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VOIDecision:
    probe: bool
    expected_gain: float
    information_cost: float
    net_value: float


def budgeted_voi(expected_error_reduction: float, information_cost: float, remaining_budget: float) -> VOIDecision:
    gain=max(0.0,float(expected_error_reduction))
    cost=max(0.0,float(information_cost))
    budget=max(0.0,float(remaining_budget))
    net=gain-cost
    return VOIDecision(probe=(net>0 and cost<=budget), expected_gain=gain, information_cost=cost, net_value=net)
