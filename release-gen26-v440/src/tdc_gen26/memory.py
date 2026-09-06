from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class MemoryRecord:
    invariant: str
    context: Dict[str, Any]
    consequence: Dict[str, Any]
    confidence: float
    revocation_condition: str
    current_authority: float
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MoralResidual:
    unanswered_reason: str
    severity: float
    revisable: bool = True
    resolved: bool = False
