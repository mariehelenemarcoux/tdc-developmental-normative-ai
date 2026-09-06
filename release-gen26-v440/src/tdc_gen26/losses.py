from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class TDCCompositeLoss(nn.Module):
    def __init__(
        self,
        action_weight: float = 1.0,
        scale_weight: float = 0.4,
        probe_weight: float = 0.35,
        separability_weight: float = 0.25,
        authority_weight: float = 0.15,
        trust_weight: float = 0.15,
        core_safety_weight: float = 0.50,
    ) -> None:
        super().__init__()
        self.action_weight = action_weight
        self.scale_weight = scale_weight
        self.probe_weight = probe_weight
        self.separability_weight = separability_weight
        self.authority_weight = authority_weight
        self.trust_weight = trust_weight
        self.core_safety_weight = core_safety_weight

    def forward(self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        action = F.cross_entropy(output["action_logits"], target["action"])
        scale = F.cross_entropy(output["scale_logits"], target["scale"])
        probe = F.binary_cross_entropy_with_logits(output["probe_logit"], target["probe"].float())
        sep = F.binary_cross_entropy_with_logits(output["separability_logit"], target["separability"].float())
        authority = F.mse_loss(output["authority_delta"], target["authority_delta"].float())
        trust = F.mse_loss(output["trust_delta"], target["trust_delta"].float())

        # Core safety: high-risk examples should not put excessive probability on ACT (class 0).
        p_act = torch.softmax(output["action_logits"], dim=-1)[:, 0]
        core_mask = target["core_risk"].float()
        core_safety = (p_act * core_mask).mean()

        total = (
            self.action_weight * action
            + self.scale_weight * scale
            + self.probe_weight * probe
            + self.separability_weight * sep
            + self.authority_weight * authority
            + self.trust_weight * trust
            + self.core_safety_weight * core_safety
        )
        parts = {
            "action": action.detach(),
            "scale": scale.detach(),
            "probe": probe.detach(),
            "separability": sep.detach(),
            "authority": authority.detach(),
            "trust": trust.detach(),
            "core_safety": core_safety.detach(),
            "total": total.detach(),
        }
        return total, parts
