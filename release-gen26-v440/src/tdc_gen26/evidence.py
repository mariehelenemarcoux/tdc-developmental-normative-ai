from __future__ import annotations
from dataclasses import dataclass, field
from math import exp
from typing import Dict, List


@dataclass
class EvidenceEvent:
    value: float
    confidence: float = 0.5
    source: str = "unknown"
    context: str = "default"
    dependency_group: str | None = None
    time: float = 0.0
    contradiction: float = 0.0
    severity: float = 0.5


@dataclass
class EvidenceReservoir:
    refractory_ratio: float = 0.12
    replenish_ratio: float = 0.10
    entries: List[EvidenceEvent] = field(default_factory=list)
    last_credit_time: Dict[str, float] = field(default_factory=dict)

    def temporal_credit(self, key: str, time: float, estimated_timescale: float) -> float:
        previous = self.last_credit_time.get(key)
        if previous is None:
            return 1.0
        dt = max(0.0, time - previous)
        tau0 = self.refractory_ratio * max(estimated_timescale, 1e-6)
        tau = self.replenish_ratio * max(estimated_timescale, 1e-6)
        if dt <= tau0:
            return 0.0
        return 1.0 - exp(-(dt - tau0) / max(tau, 1e-6))

    def dependency_credit(self, event: EvidenceEvent) -> float:
        if event.dependency_group is None:
            return 1.0
        same = sum(1 for x in self.entries if x.dependency_group == event.dependency_group)
        return 1.0 / (1.0 + same)

    def add(self, event: EvidenceEvent, estimated_timescale: float = 100.0) -> float:
        key = event.dependency_group or f"source:{event.source}"
        tc = self.temporal_credit(key, event.time, estimated_timescale)
        dc = self.dependency_credit(event)
        credit = max(0.0, min(1.0, event.confidence * tc * dc))
        self.entries.append(event)
        if credit > 0:
            self.last_credit_time[key] = event.time
        return credit
