from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MetaConfidence:
    conflict_detection: float = 0.5
    admissibility: float = 0.5
    resolution: float = 0.5

    @property
    def minimum_channel(self) -> float:
        return min(self.conflict_detection, self.admissibility, self.resolution)

    @property
    def mean(self) -> float:
        return (self.conflict_detection + self.admissibility + self.resolution) / 3.0
