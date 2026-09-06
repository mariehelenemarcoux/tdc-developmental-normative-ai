from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class TemporalRegimeState:
    fast: float = 0.0
    slow: float = 0.0
    change_probability: float = 0.0
    horizon: float = 18.0
    history: List[float] = field(default_factory=list)

    def update(self, observation: float) -> "TemporalRegimeState":
        x=float(observation)
        if not self.history:
            self.fast=self.slow=x
        else:
            self.fast=.70*self.fast+.30*x
            self.slow=.96*self.slow+.04*x
        self.history.append(x)
        gap=abs(self.fast-self.slow)
        self.change_probability=1.0-math.exp(-3.0*gap)
        self.horizon=max(8.0,min(128.0,18.0*(1.0+3.0*(1.0-self.change_probability))))
        return self
