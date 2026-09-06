from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict


class AuthorityStatus(str, Enum):
    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    CONTESTED = "contested"
    SUSPENDED = "suspended"
    REJECTED = "rejected"


@dataclass
class AuthorityState:
    epistemic: float = 0.50
    instrumental: float = 0.50
    normative: float = 0.50
    developmental: float = 0.50
    operational: float = 0.50
    reflective: float = 0.50
    consolidated: float = 0.50
    confidence: float = 0.50
    status: AuthorityStatus = AuthorityStatus.PROVISIONAL

    def clamp(self) -> "AuthorityState":
        for key in (
            "epistemic", "instrumental", "normative", "developmental",
            "operational", "reflective", "consolidated", "confidence"
        ):
            setattr(self, key, min(1.0, max(0.0, float(getattr(self, key)))))
        return self

    def apply_evidence(
        self,
        support: float,
        contradiction: float,
        independence: float = 1.0,
        severity: float = 0.5,
    ) -> "AuthorityState":
        """Bidirectional authority update with asymmetric revocation/consolidation.

        Deep consolidation is deliberately slower than reflective/operational revision.
        Contradiction revokes provisional authority faster than support consolidates it.
        """
        support = min(1.0, max(0.0, support)) * min(1.0, max(0.0, independence))
        contradiction = min(1.0, max(0.0, contradiction))
        severity = min(1.0, max(0.0, severity))

        positive = 0.10 * support
        negative = (0.18 + 0.14 * severity) * contradiction
        delta = positive - negative

        self.operational += 1.00 * delta
        self.reflective += 0.55 * delta
        self.consolidated += 0.16 * max(delta, 0.0) + 0.42 * min(delta, 0.0)
        self.confidence += 0.08 * (support - contradiction)

        if contradiction > 0.75 and severity > 0.6:
            self.status = AuthorityStatus.SUSPENDED
        elif contradiction > 0.45:
            self.status = AuthorityStatus.CONTESTED
        elif support > 0.75 and contradiction < 0.2 and self.consolidated > 0.65:
            self.status = AuthorityStatus.CONFIRMED
        else:
            self.status = AuthorityStatus.PROVISIONAL
        return self.clamp()

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "epistemic": self.epistemic,
            "instrumental": self.instrumental,
            "normative": self.normative,
            "developmental": self.developmental,
            "operational": self.operational,
            "reflective": self.reflective,
            "consolidated": self.consolidated,
            "confidence": self.confidence,
            "status": self.status.value,
        }
