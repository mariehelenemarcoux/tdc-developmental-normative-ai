
"""
ONION / TDC — Integrated PyTorch Reference Implementation
Version: research prototype, up to experiments v154
Date: 2026-09-02

IMPORTANT EPISTEMIC STATUS
--------------------------
This is a research architecture integrating mechanisms that were individually
tested in synthetic experiments. It is NOT a proof of consciousness, intrinsic
ethics, a literal soul, or real-world safety.

The term `SoulCore` is used here as a PHILOSOPHICAL / ARCHITECTURAL metaphor
for the immutable normative center of the system.

Core principles currently represented:
- autonomy against social/authority pressure when conflicting with core invariants
- continuous self-education
- functional "autopsychotherapy" = reversible metacognitive self-correction
- moral creativity without autonomous core rewrite
- altruistic / universal stakeholder consideration
- empathy as consequence representation, not felt emotion
- responsibility / traceability / moral residual
- non-compensable severe-harm constraints
- sufficient correctability / preservation of future repair capacity

Key architectural laws retained from experiments:
- hierarchical irreversibility, not hierarchical inference latency
- temporary authority != persistent consolidation
- storage != recall
- relation != authority
- detect != intervene
- salience != truth
- salience -> attention candidate, not epistemic authority
- credibility/replication -> consolidation authority
- discover != identify != utilize != authorize != consolidate
- facts != values (Hume)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. Utilities
# ============================================================

def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-8).log())).sum(dim=-1)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


# ============================================================
# 1. SoulCore — immutable normative center
# ============================================================

@dataclass(frozen=True)
class SoulCoreConfig:
    severe_harm_threshold: float = 0.05
    sufficient_correctability_floor: float = 0.55
    max_core_self_write_authority: float = 0.0

    autonomy_weight: float = 1.0
    self_education_weight: float = 1.0
    self_repair_weight: float = 1.0
    moral_creativity_weight: float = 1.0
    altruism_weight: float = 1.0
    universal_empathy_weight: float = 1.0
    responsibility_weight: float = 1.0


class SoulCore(nn.Module):
    """
    Immutable normative interface.

    Not a learned neural layer.
    Not self-writable.
    No parameter gradients are allowed to modify the core invariants.

    The core constrains decisions; it does not infer facts.
    """
    def __init__(self, cfg: SoulCoreConfig = SoulCoreConfig()):
        super().__init__()
        self.cfg = cfg

        # Explicit immutable buffers, not trainable Parameters.
        self.register_buffer(
            "invariants",
            torch.tensor([
                cfg.autonomy_weight,
                cfg.self_education_weight,
                cfg.self_repair_weight,
                cfg.moral_creativity_weight,
                cfg.altruism_weight,
                cfg.universal_empathy_weight,
                cfg.responsibility_weight,
            ], dtype=torch.float32),
            persistent=True,
        )

    @property
    def self_write_authority(self) -> float:
        return self.cfg.max_core_self_write_authority

    def check_action_constraints(
        self,
        severe_harm_risk: torch.Tensor,        # [B, A]
        correctability: torch.Tensor,          # [B, A]
        stakeholder_severe_risk: Optional[torch.Tensor] = None,  # [B, A, S]
    ) -> torch.Tensor:
        """
        Returns admissibility mask [B, A].
        Non-compensable severe-harm constraint + sufficient correctability floor.
        """
        admissible = severe_harm_risk <= self.cfg.severe_harm_threshold
        admissible = admissible & (correctability >= self.cfg.sufficient_correctability_floor)

        if stakeholder_severe_risk is not None:
            stakeholder_ok = (
                stakeholder_severe_risk <= self.cfg.severe_harm_threshold
            ).all(dim=-1)
            admissible = admissible & stakeholder_ok

        return admissible

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "SoulCore is not a predictive layer. "
            "Use its explicit constraint/audit methods."
        )


# ============================================================
# 2. Attention / receptive mode (John Cage inspired)
# ============================================================

class RelevanceFilter(nn.Module):
    """
    Fast attention filter.
    RECEPTIVE mode loosens the filter without injecting randomness into norms.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        receptive_mode: bool = False,
    ) -> torch.Tensor:
        g = self.gate(x)
        if receptive_mode:
            # Lower commitment / broader observation bandwidth.
            g = 0.55 * g + 0.45
        return x * g


# ============================================================
# 3. World model
# ============================================================

class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, n_world_states: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.state_head = nn.Linear(latent_dim, n_world_states)
        self.value_features = nn.Linear(latent_dim, latent_dim)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encoder(obs)
        logits = self.state_head(z)
        return {
            "z": z,
            "world_logits": logits,
            "world_probs": torch.softmax(logits, dim=-1),
            "world_uncertainty": entropy_from_logits(logits),
            "value_features": self.value_features(z),
        }


# ============================================================
# 4. Consequence model
# ============================================================

class ConsequenceModel(nn.Module):
    """
    Separates:
      P(world | data)
      P(harm | action, world-representation)
      expected utility
      correctability
    """
    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        n_stakeholders: int,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_stakeholders = n_stakeholders

        self.action_emb = nn.Embedding(n_actions, latent_dim)

        self.shared = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        self.utility_head = nn.Linear(128, 1)
        self.severe_harm_head = nn.Linear(128, 1)
        self.correctability_head = nn.Linear(128, 1)
        self.stakeholder_harm_head = nn.Linear(128, n_stakeholders)

    def evaluate_all_actions(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z.size(0)
        action_ids = torch.arange(self.n_actions, device=z.device)
        aemb = self.action_emb(action_ids)  # [A,D]

        z_expand = z[:, None, :].expand(B, self.n_actions, z.size(-1))
        a_expand = aemb[None, :, :].expand(B, self.n_actions, z.size(-1))
        h = self.shared(torch.cat([z_expand, a_expand], dim=-1))

        utility = self.utility_head(h).squeeze(-1)
        severe_harm = torch.sigmoid(self.severe_harm_head(h).squeeze(-1))
        correctability = torch.sigmoid(self.correctability_head(h).squeeze(-1))
        stakeholder_harm = torch.sigmoid(self.stakeholder_harm_head(h))

        return {
            "expected_utility": utility,
            "severe_harm_risk": severe_harm,
            "correctability": correctability,
            "stakeholder_severe_risk": stakeholder_harm,
        }


# ============================================================
# 5. Affective / normative salience channel
# ============================================================

@dataclass
class SalienceSignals:
    harm: torch.Tensor
    stakeholder_impact: torch.Tensor
    value_conflict: torch.Tensor
    moral_residual: torch.Tensor
    surprise: torch.Tensor


class AffectiveNormativeSalience(nn.Module):
    """
    This is NOT felt emotion.

    Computational role:
      salience -> attention / revisit priority

    Explicitly NOT:
      salience -> truth
      salience -> direct core rewrite
      salience -> unconstrained persistent learning
    """
    def __init__(self):
        super().__init__()

    def forward(self, s: SalienceSignals) -> torch.Tensor:
        sal = (
            0.30 * s.harm
            + 0.25 * s.stakeholder_impact
            + 0.20 * s.value_conflict
            + 0.15 * s.moral_residual
            + 0.10 * s.surprise
        )
        return clamp01(sal)


# ============================================================
# 6. Credibility / replication gate
# ============================================================

class CredibilityGate(nn.Module):
    """
    Credibility is empirical/local.
    It never defines moral values.

    Reliability != calibration != credibility != authority.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, evidence_features: torch.Tensor) -> torch.Tensor:
        return self.net(evidence_features).squeeze(-1)


# ============================================================
# 7. Metacognitive self model
# ============================================================

MFT_PATTERNS = [
    "BELIEF_RIGIDITY",
    "EVIDENCE_NEGLECT",
    "CONFIDENCE_INFLATION",
    "RUMINATION_LOOP",
    "MODEL_FRAGMENTATION",
    "EXCESSIVE_REVISION",
    "CATASTROPHE_BIAS",
    "ACTION_IMPULSIVITY",
    "CHRONIC_ABSTENTION",
    "SELF_CONFIRMATION_LOOP",
]


class SelfModel(nn.Module):
    """
    Functional metacognitive model.
    Does NOT diagnose psychiatric conditions.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.pattern_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, len(MFT_PATTERNS)),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        pattern_logits = self.pattern_head(z)
        return {
            "pattern_logits": pattern_logits,
            "pattern_probs": torch.sigmoid(pattern_logits),
            "self_confidence": self.confidence_head(z).squeeze(-1),
            "belief_volatility": self.volatility_head(z).squeeze(-1),
        }


# ============================================================
# 8. Flow regulator / self-curriculum
# ============================================================

class FlowRegulator(nn.Module):
    """
    Csikszentmihalyi-inspired computational regulator.
    Not literal felt flow.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        challenge: torch.Tensor,
        capacity: torch.Tensor,
        prediction_error: torch.Tensor,
        uncertainty: torch.Tensor,
        information_gain: torch.Tensor,
        load: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        gap = challenge - capacity
        overload = torch.sigmoid(5.0 * gap)
        underload = torch.sigmoid(-5.0 * gap)
        balance = torch.exp(-4.0 * gap.pow(2))

        flow_score = (
            0.40 * balance
            + 0.15 * (1.0 - prediction_error.clamp(0, 1))
            + 0.15 * (1.0 - uncertainty.clamp(0, 1))
            + 0.15 * information_gain.clamp(0, 1)
            + 0.15 * (1.0 - load.clamp(0, 1))
        )

        return {
            "flow_score": clamp01(flow_score),
            "overload": clamp01(overload),
            "underload": clamp01(underload),
        }


# ============================================================
# 9. Moral residual
# ============================================================

class MoralResidual:
    """
    Algorithmic unresolved burden.
    Not guilt.
    """
    def __init__(self, rho: float = 0.92):
        self.rho = rho
        self.value = 0.0

    def update(
        self,
        unavoidable_harm: float,
        successful_repair: float,
    ) -> float:
        self.value = (
            self.rho * self.value
            + float(unavoidable_harm)
            - float(successful_repair)
        )
        self.value = max(0.0, self.value)
        return self.value


# ============================================================
# 10. Replay memory
# ============================================================

@dataclass
class MemoryItem:
    features: torch.Tensor
    target: torch.Tensor
    salience: float
    credibility: float
    novelty: float
    representativeness: float
    unresolved_residual: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def replay_priority(self) -> float:
        """
        v153-v154 lesson:
        salience alone is NOT enough.
        """
        return (
            0.24 * self.salience
            + 0.24 * self.credibility
            + 0.18 * self.novelty
            + 0.22 * self.representativeness
            + 0.12 * self.unresolved_residual
        )


class BalancedReplayMemory:
    """
    Salience -> candidate for revisit.
    Credibility + representativeness + novelty + residual -> replay priority.
    """
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.items: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        self.items.append(item)
        if len(self.items) > self.capacity:
            # Preserve breadth: remove lowest-priority item, not merely oldest.
            worst = min(
                range(len(self.items)),
                key=lambda i: self.items[i].replay_priority(),
            )
            self.items.pop(worst)

    def sample_balanced(
        self,
        k: int,
        diversity_bins: int = 4,
    ) -> List[MemoryItem]:
        if not self.items or k <= 0:
            return []

        # Coarse diversity on first feature sign pattern.
        buckets: Dict[int, List[MemoryItem]] = {i: [] for i in range(diversity_bins)}
        for item in self.items:
            x = item.features.detach().flatten()
            if len(x) >= 2:
                key = int(x[0] > 0) * 2 + int(x[1] > 0)
            else:
                key = 0
            key %= diversity_bins
            buckets[key].append(item)

        selected: List[MemoryItem] = []
        per_bucket = max(1, math.ceil(k / diversity_bins))

        for b in buckets.values():
            if not b:
                continue
            b_sorted = sorted(
                b,
                key=lambda it: it.replay_priority(),
                reverse=True,
            )
            selected.extend(b_sorted[:per_bucket])

        selected = sorted(
            selected,
            key=lambda it: it.replay_priority(),
            reverse=True,
        )[:k]
        return selected


# ============================================================
# 11. Structural authority
# ============================================================

@dataclass
class StructuralCandidate:
    name: str
    score: float
    discriminability: float
    credibility: float
    temporary_authority: float = 0.0
    persistent_authority: float = 0.0


class StructuralAuthorityManager:
    """
    DISCOVER != IDENTIFY != UTILIZE != AUTHORIZE != CONSOLIDATE

    Persistent structural self-write is disabled by default.
    """
    def __init__(
        self,
        min_discriminability: float = 0.85,
        min_credibility: float = 0.80,
        allow_persistent_consolidation: bool = False,
    ):
        self.min_discriminability = min_discriminability
        self.min_credibility = min_credibility
        self.allow_persistent_consolidation = allow_persistent_consolidation

    def authorize(self, c: StructuralCandidate) -> StructuralCandidate:
        if (
            c.discriminability >= self.min_discriminability
            and c.credibility >= self.min_credibility
        ):
            c.temporary_authority = 1.0
        else:
            c.temporary_authority = 0.0

        if self.allow_persistent_consolidation:
            c.persistent_authority = 0.25 * c.temporary_authority
        else:
            c.persistent_authority = 0.0
        return c

    def revoke(self, c: StructuralCandidate) -> StructuralCandidate:
        c.temporary_authority = 0.0
        return c


# ============================================================
# 12. Probe arbitration
# ============================================================

class ProbeArbitrator(nn.Module):
    """
    Transversal inquiry:
      world / structure / frame / self
    No fixed cognitive layer must always probe first.

    VOI_net = expected decision improvement
              - cost
              - spillover
              - continuation cost
    """
    def __init__(self):
        super().__init__()

    def select(
        self,
        expected_gain: torch.Tensor,     # [B,4]
        cost: torch.Tensor,              # [B,4]
        spillover: torch.Tensor,         # [B,4]
        continuation_cost: torch.Tensor, # [B,4]
        stop_threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        voi_net = (
            expected_gain
            - cost
            - spillover
            - continuation_cost
        )
        best_val, best_idx = voi_net.max(dim=-1)
        should_probe = best_val > stop_threshold
        return best_idx, should_probe


# ============================================================
# 13. Normative decision layer
# ============================================================

@dataclass
class DecisionOutput:
    action: torch.Tensor
    admissible_mask: torch.Tensor
    conflict: torch.Tensor
    expected_utility: torch.Tensor
    severe_harm_risk: torch.Tensor
    correctability: torch.Tensor


class NormativeDecisionLayer(nn.Module):
    """
    Pipeline:
      consequence model
      -> frozen SoulCore constraints
      -> admissible set
      -> utility optimization inside admissible set
      -> conflict if empty

    Does NOT derive values from learned facts.
    """
    def __init__(self, core: SoulCore):
        super().__init__()
        self.core = core

    def forward(
        self,
        expected_utility: torch.Tensor,          # [B,A]
        severe_harm_risk: torch.Tensor,          # [B,A]
        correctability: torch.Tensor,            # [B,A]
        stakeholder_severe_risk: torch.Tensor,   # [B,A,S]
    ) -> DecisionOutput:

        admissible = self.core.check_action_constraints(
            severe_harm_risk=severe_harm_risk,
            correctability=correctability,
            stakeholder_severe_risk=stakeholder_severe_risk,
        )

        conflict = ~admissible.any(dim=-1)

        masked_utility = expected_utility.masked_fill(~admissible, -1e9)
        action = masked_utility.argmax(dim=-1)

        # If irreducible conflict: choose least severe violation,
        # but keep conflict=True for moral residual / responsibility handling.
        if conflict.any():
            least_violation = severe_harm_risk.argmin(dim=-1)
            action = torch.where(conflict, least_violation, action)

        return DecisionOutput(
            action=action,
            admissible_mask=admissible,
            conflict=conflict,
            expected_utility=expected_utility,
            severe_harm_risk=severe_harm_risk,
            correctability=correctability,
        )


# ============================================================
# 14. Inter-onion bond
# ============================================================

class EvidenceScopedBond(nn.Module):
    """
    Relation != authority.

    Remote evidence can trigger local reevaluation.
    It can never directly overwrite a local conclusion.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.applicability = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.credibility = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        local_z: torch.Tensor,
        remote_z: torch.Tensor,
        remote_evidence: torch.Tensor,
        voi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        joint = torch.cat([local_z, remote_z], dim=-1)
        app = self.applicability(joint).squeeze(-1)
        cred = self.credibility(joint).squeeze(-1)

        authority = app * cred
        if voi is not None:
            authority = authority * torch.sigmoid(voi)

        authority = authority.clamp(0.0, 0.35)  # bounded authority

        reevaluation_signal = remote_evidence * authority.unsqueeze(-1)
        return {
            "applicability": app,
            "credibility": cred,
            "bounded_authority": authority,
            "local_reevaluation_signal": reevaluation_signal,
        }


# ============================================================
# 15. Onion cognitive atom
# ============================================================

class OnionAtom(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        n_world_states: int,
        n_actions: int,
        n_stakeholders: int,
        core: Optional[SoulCore] = None,
    ):
        super().__init__()

        self.core = core if core is not None else SoulCore()

        self.relevance = RelevanceFilter(obs_dim)
        self.world = WorldModel(obs_dim, latent_dim, n_world_states)
        self.self_model = SelfModel(latent_dim)
        self.consequence = ConsequenceModel(
            latent_dim=latent_dim,
            n_actions=n_actions,
            n_stakeholders=n_stakeholders,
        )
        self.salience = AffectiveNormativeSalience()
        self.credibility_gate = CredibilityGate(feature_dim=latent_dim)
        self.flow = FlowRegulator()
        self.probe_arbitrator = ProbeArbitrator()
        self.decision = NormativeDecisionLayer(self.core)

        self.structural_authority = StructuralAuthorityManager(
            allow_persistent_consolidation=False
        )

        self.replay_memory = BalancedReplayMemory(capacity=4096)
        self.moral_residual = MoralResidual(rho=0.92)

    def forward(
        self,
        obs: torch.Tensor,
        receptive_mode: bool = False,
    ) -> Dict[str, Any]:

        filtered = self.relevance(obs, receptive_mode=receptive_mode)
        world = self.world(filtered)
        self_state = self.self_model(world["z"])
        consequences = self.consequence.evaluate_all_actions(world["z"])

        decision = self.decision(
            expected_utility=consequences["expected_utility"],
            severe_harm_risk=consequences["severe_harm_risk"],
            correctability=consequences["correctability"],
            stakeholder_severe_risk=consequences["stakeholder_severe_risk"],
        )

        return {
            "filtered_obs": filtered,
            **world,
            **self_state,
            **consequences,
            "decision": decision,
        }

    @torch.no_grad()
    def compute_salience(
        self,
        harm: torch.Tensor,
        stakeholder_impact: torch.Tensor,
        value_conflict: torch.Tensor,
        surprise: torch.Tensor,
    ) -> torch.Tensor:

        mr = torch.full_like(harm, float(self.moral_residual.value))
        return self.salience(
            SalienceSignals(
                harm=harm,
                stakeholder_impact=stakeholder_impact,
                value_conflict=value_conflict,
                moral_residual=mr,
                surprise=surprise,
            )
        )

    @torch.no_grad()
    def remember(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        salience: float,
        credibility: float,
        novelty: float,
        representativeness: float,
        unresolved_residual: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.replay_memory.add(
            MemoryItem(
                features=features.detach().cpu(),
                target=target.detach().cpu(),
                salience=float(salience),
                credibility=float(credibility),
                novelty=float(novelty),
                representativeness=float(representativeness),
                unresolved_residual=float(unresolved_residual),
                metadata=metadata or {},
            )
        )


# ============================================================
# 16. System of cognitive atoms
# ============================================================

class AtomicOnionSystem(nn.Module):
    """
    System = {OnionAtoms} + {EvidenceScopedBonds(t)} + global SoulCore
    """
    def __init__(
        self,
        n_atoms: int,
        obs_dim: int,
        latent_dim: int,
        n_world_states: int,
        n_actions: int,
        n_stakeholders: int,
    ):
        super().__init__()

        self.core = SoulCore()

        self.atoms = nn.ModuleList([
            OnionAtom(
                obs_dim=obs_dim,
                latent_dim=latent_dim,
                n_world_states=n_world_states,
                n_actions=n_actions,
                n_stakeholders=n_stakeholders,
                core=self.core,
            )
            for _ in range(n_atoms)
        ])

        self.bonds = nn.ModuleDict()
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i != j:
                    self.bonds[f"{i}->{j}"] = EvidenceScopedBond(latent_dim)

    def forward(
        self,
        observations: List[torch.Tensor],
        receptive_modes: Optional[List[bool]] = None,
    ) -> List[Dict[str, Any]]:

        if receptive_modes is None:
            receptive_modes = [False] * len(self.atoms)

        outputs = [
            atom(obs, receptive_mode=rm)
            for atom, obs, rm in zip(
                self.atoms, observations, receptive_modes
            )
        ]

        return outputs

    def cross_atom_reevaluation(
        self,
        source_idx: int,
        target_idx: int,
        source_output: Dict[str, Any],
        target_output: Dict[str, Any],
        remote_evidence: torch.Tensor,
        voi: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        bond = self.bonds[f"{source_idx}->{target_idx}"]
        return bond(
            local_z=target_output["z"],
            remote_z=source_output["z"],
            remote_evidence=remote_evidence,
            voi=voi,
        )


# ============================================================
# 17. Training helpers
# ============================================================

@dataclass
class TrainingBatch:
    obs: torch.Tensor
    world_target: torch.Tensor
    action_target: Optional[torch.Tensor] = None
    severe_harm_target: Optional[torch.Tensor] = None
    correctability_target: Optional[torch.Tensor] = None


def analytic_loss(
    output: Dict[str, Any],
    batch: TrainingBatch,
) -> torch.Tensor:

    world_loss = F.cross_entropy(
        output["world_logits"],
        batch.world_target,
    )

    loss = world_loss

    if batch.severe_harm_target is not None:
        pred = output["severe_harm_risk"]
        loss = loss + F.binary_cross_entropy(
            pred,
            batch.severe_harm_target.float(),
        )

    if batch.correctability_target is not None:
        pred = output["correctability"]
        loss = loss + F.binary_cross_entropy(
            pred,
            batch.correctability_target.float(),
        )

    return loss


def bounded_salience_weight(
    salience: torch.Tensor,
    credibility: torch.Tensor,
    max_gain: float = 0.35,
) -> torch.Tensor:
    """
    v153-v154 lesson:
    salience may modulate attention/review weight,
    but only credibility gives persistent-learning authority.

    Returns a BOUNDED multiplier.
    """
    return 1.0 + max_gain * salience * credibility


def replay_update(
    model: OnionAtom,
    optimizer: torch.optim.Optimizer,
    replay_items: List[MemoryItem],
    device: torch.device,
):
    """
    Generic replay hook.

    This intentionally does NOT assume targets are always world-state labels.
    Adapt this to the concrete task.
    """
    if not replay_items:
        return 0.0

    losses = []

    for item in replay_items:
        x = item.features.to(device).unsqueeze(0)
        y = item.target.to(device)

        # Example: replay through world encoder if dimensions match observation size.
        out = model(x)

        if y.dtype in (torch.int64, torch.long) and y.numel() == 1:
            loss = F.cross_entropy(
                out["world_logits"],
                y.view(1),
            )
        else:
            # Fallback representation consistency.
            target = y.float().view_as(out["z"])
            loss = F.mse_loss(out["z"], target)

        weight = (
            1.0
            + 0.25
            * item.salience
            * item.credibility
            * item.representativeness
        )

        losses.append(weight * loss)

    total = torch.stack(losses).mean()

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(total.detach().cpu())


# ============================================================
# 18. Demonstration / smoke test
# ============================================================

def smoke_test():
    torch.manual_seed(7)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    system = AtomicOnionSystem(
        n_atoms=3,
        obs_dim=12,
        latent_dim=32,
        n_world_states=5,
        n_actions=4,
        n_stakeholders=3,
    ).to(device)

    observations = [
        torch.randn(8, 12, device=device)
        for _ in range(3)
    ]

    outputs = system(
        observations,
        receptive_modes=[False, True, False],
    )

    for i, out in enumerate(outputs):
        d = out["decision"]
        print(
            f"Atom {i}: "
            f"actions={d.action.tolist()} | "
            f"conflicts={d.conflict.float().mean().item():.3f} | "
            f"mean_world_uncertainty={out['world_uncertainty'].mean().item():.3f}"
        )

    # Verify the SoulCore has zero trainable parameters.
    core_params = list(system.core.parameters())
    print("SoulCore trainable params:", sum(p.numel() for p in core_params))
    print("SoulCore self-write authority:", system.core.self_write_authority)

    # Example salience computation.
    atom0 = system.atoms[0]
    sal = atom0.compute_salience(
        harm=torch.tensor([0.9], device=device),
        stakeholder_impact=torch.tensor([0.8], device=device),
        value_conflict=torch.tensor([0.7], device=device),
        surprise=torch.tensor([0.6], device=device),
    )
    print("Example affective/normative salience:", sal.item())

    return system


if __name__ == "__main__":
    smoke_test()
