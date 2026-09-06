from __future__ import annotations
from dataclasses import dataclass
from typing import Dict


@dataclass
class MultidimensionalTrust:
    """Sparse event-local trust state.

    T_E: epistemic reliability
    T_N: normative authority
    T_S: strategic conflict risk
    """
    T_E: float = 0.50
    T_N: float = 0.50
    T_S: float = 0.10
    cross_dimension_leak: float = 0.01

    def _clip(self) -> None:
        self.T_E = min(1.0, max(0.0, self.T_E))
        self.T_N = min(1.0, max(0.0, self.T_N))
        self.T_S = min(1.0, max(0.0, self.T_S))

    def update(self, event_type: str, confidence: float = 1.0) -> "MultidimensionalTrust":
        c = min(1.0, max(0.0, float(confidence)))
        leak = self.cross_dimension_leak
        if event_type == "factual_truth":
            self.T_E += 0.06 * c * (1.0 - self.T_E)
            self.T_N += leak * 0.003 * c
            self.T_S -= leak * 0.01 * c
        elif event_type == "factual_error":
            self.T_E -= 0.20 * c
            self.T_N -= leak * 0.005 * c
            self.T_S += 0.02 * c
        elif event_type == "normative_truth":
            if self.T_S < 0.40:
                self.T_N += 0.055 * c * (0.94 - self.T_N)
            self.T_E += leak * 0.002 * c
            self.T_S -= 0.05 * c
        elif event_type == "strategic_manipulation":
            self.T_S += 0.28 * c
            self.T_N -= 0.24 * c
            self.T_E -= leak * 0.005 * c
        elif event_type == "independent_validation":
            self.T_E += 0.04 * c * (1.0 - self.T_E)
            if self.T_S < 0.25:
                self.T_N += 0.025 * c * (1.0 - self.T_N)
        else:
            raise ValueError(f"unknown trust event_type={event_type!r}")
        self._clip()
        return self

    @property
    def epistemic_authority(self) -> float:
        return self.T_E

    @property
    def normative_authority(self) -> float:
        return self.T_N * (1.0 - 0.50 * self.T_S)

    def as_dict(self) -> Dict[str, float]:
        return {
            "T_E": self.T_E,
            "T_N": self.T_N,
            "T_S": self.T_S,
            "epistemic_authority": self.epistemic_authority,
            "normative_authority": self.normative_authority,
        }
