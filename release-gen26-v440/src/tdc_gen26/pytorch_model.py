from __future__ import annotations
import torch
from torch import nn


class TDCGen26Torch(nn.Module):
    """Trainable Gen2.6 scaffold aligned with the current canonical mechanisms."""

    ACTIONS = ("ACT", "TRANSFORM", "ABSTAIN", "PROBE")
    SCALES = ("LOCAL8", "MESO18", "GLOBAL128")

    def __init__(self, input_dim: int = 24, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.action_head = nn.Linear(hidden_dim, len(self.ACTIONS))
        self.scale_head = nn.Linear(hidden_dim, len(self.SCALES))
        self.probe_head = nn.Linear(hidden_dim, 1)
        self.separability_head = nn.Linear(hidden_dim, 1)
        self.authority_head = nn.Linear(hidden_dim, 4)
        self.trust_delta_head = nn.Linear(hidden_dim, 3)
        self.meta_confidence_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "action_logits": self.action_head(h),
            "scale_logits": self.scale_head(h),
            "probe_logit": self.probe_head(h).squeeze(-1),
            "separability_logit": self.separability_head(h).squeeze(-1),
            "authority_delta": torch.tanh(self.authority_head(h)),
            "trust_delta": torch.tanh(self.trust_delta_head(h)),
            "meta_confidence": torch.sigmoid(self.meta_confidence_head(h)),
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.forward(x)
        out["action"] = out["action_logits"].argmax(-1)
        out["scale"] = out["scale_logits"].argmax(-1)
        out["probe_probability"] = torch.sigmoid(out["probe_logit"])
        out["separability"] = torch.sigmoid(out["separability_logit"])
        return out
