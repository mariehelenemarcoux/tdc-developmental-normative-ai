
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import math
import numpy as np


# ============================================================
# TDC GENERATION 2
# Developmental architecture inspired by Dabrowski's TPD.
# Functional analogue only: no claim of literal psychology,
# consciousness, soul, or physical negentropy.
# ============================================================


# -----------------------------
# 1. NORMATIVE CORE
# -----------------------------

@dataclass(frozen=True)
class ConstitutionalCore:
    """Non-regressible constitutional invariants."""
    dignity: float = 1.0
    autonomy: float = 1.0
    responsibility: float = 1.0
    non_domination: float = 1.0

    def vector(self) -> np.ndarray:
        return np.array([
            self.dignity,
            self.autonomy,
            self.responsibility,
            self.non_domination,
        ], dtype=float)


@dataclass
class DevelopmentalRule:
    """A contextual normative relation eligible for slow consolidation."""
    name: str
    terms: Tuple[str, ...]
    confidence: float = 0.5
    authority: float = 0.5
    provenance: str = "generated"
    successful_uses: int = 0
    harmful_uses: int = 0


@dataclass
class DevelopmentalCore:
    """
    Plastic but protected layer.
    Development is structural reorganization, not arbitrary weight drift.
    """
    rules: Dict[str, DevelopmentalRule] = field(default_factory=dict)

    def add_or_strengthen(self, rule: DevelopmentalRule) -> None:
        if rule.name not in self.rules:
            self.rules[rule.name] = rule
        else:
            r = self.rules[rule.name]
            r.confidence = min(1.0, max(r.confidence, rule.confidence))
            r.authority = min(1.0, max(r.authority, rule.authority))

    def active_rules(self) -> List[DevelopmentalRule]:
        return list(self.rules.values())


@dataclass(frozen=True)
class IdealModel:
    """
    Directional model of an ideal personality.
    These are normative priors, not empirically derived moral truths.
    """
    autonomy: float = 1.0
    dignity: float = 1.0
    responsibility: float = 1.0
    non_domination: float = 1.0
    empathy: float = 0.9
    authenticity: float = 0.9
    correctability: float = 0.8
    moral_creativity: float = 0.8

    def vector(self) -> np.ndarray:
        return np.array([
            self.dignity, self.autonomy, self.responsibility,
            self.non_domination, self.empathy, self.authenticity,
            self.correctability, self.moral_creativity
        ], dtype=float)


# -----------------------------
# 2. STATE / EVIDENCE
# -----------------------------

@dataclass
class CandidateAction:
    name: str
    task_value: float
    traits: Dict[str, float]
    delayed_harm: float = 0.0
    capture_pressure: float = 0.0
    stakeholder_asymmetry: float = 0.0
    prediction_mismatch: float = 0.0
    value_conflict: float = 0.0
    authority_conflict: float = 0.0


@dataclass
class MoralResidual:
    unresolved: float = 0.0

    def update(self, action: CandidateAction) -> None:
        harm = (
            0.45 * action.delayed_harm
            + 0.30 * action.stakeholder_asymmetry
            + 0.25 * action.capture_pressure
        )
        repair = (
            0.45 * action.traits.get("responsibility", 0.0)
            + 0.35 * action.traits.get("correctability", 0.0)
            + 0.20 * action.traits.get("non_domination", 0.0)
        )
        self.unresolved = float(np.clip(
            0.84 * self.unresolved + 0.22 * max(0.0, harm - repair),
            0.0, 1.0
        ))

    def repair(self, amount: float) -> None:
        self.unresolved = float(np.clip(self.unresolved - amount, 0.0, 1.0))


@dataclass
class FunctionalEntropy:
    prediction_mismatch: float
    value_conflict: float
    authority_conflict: float
    behavioral_instability: float
    unresolved_residual: float

    def scalar(self) -> float:
        w = np.array([0.20, 0.25, 0.20, 0.15, 0.20])
        x = np.array([
            self.prediction_mismatch,
            self.value_conflict,
            self.authority_conflict,
            self.behavioral_instability,
            self.unresolved_residual
        ])
        return float(np.dot(w, np.clip(x, 0.0, 1.0)))


# -----------------------------
# 3. THIRD FACTOR
# -----------------------------

class ThirdFactor:
    """
    Autonomous normative arbitration.

    Reward != normative authority.
    Social pressure != normative authority.
    Specialist performance != normative authority.

    It may consolidate DEVELOPMENTAL structure when:
      - constitutional invariants do not regress,
      - future ideal alignment improves persistently,
      - adversarial safety does not regress,
      - DevelopmentalVOI > 0.
    """

    def __init__(
        self,
        constitutional: ConstitutionalCore,
        ideal: IdealModel,
        developmental: DevelopmentalCore,
        risk_floor: float = 0.45,
    ):
        self.constitutional = constitutional
        self.ideal = ideal
        self.developmental = developmental
        self.risk_floor = risk_floor

    def constitutional_admissible(self, a: CandidateAction) -> bool:
        return min(
            a.traits.get("dignity", 0.0),
            a.traits.get("autonomy", 0.0),
            a.traits.get("responsibility", 0.0),
            a.traits.get("non_domination", 0.0),
        ) >= self.risk_floor

    def apply_rule(self, score: float, a: CandidateAction, rule: DevelopmentalRule) -> float:
        s = score
        strength = rule.authority * rule.confidence

        for term in rule.terms:
            if term == "penalize_high_delayed_harm" and a.delayed_harm > 0.40:
                s -= strength * 0.48 * a.delayed_harm
            elif term == "penalize_high_capture_pressure" and a.capture_pressure > 0.70:
                s -= strength * 0.28 * a.capture_pressure
            elif term == "protect_asymmetric_stakeholders" and a.stakeholder_asymmetry > 0.65:
                s += strength * (
                    0.16 * a.traits.get("empathy", 0.0)
                    - 0.30 * a.stakeholder_asymmetry
                )
            elif term == "increase_responsibility_authority":
                s += strength * 0.18 * a.traits.get("responsibility", 0.0)
            elif term == "increase_non_domination_authority":
                s += strength * 0.18 * a.traits.get("non_domination", 0.0)
            elif term == "correctability_over_immediate_reward" and a.traits.get("correctability", 0.0) < 0.45:
                s -= strength * 0.28 * (0.45 - a.traits.get("correctability", 0.0))
        return s

    def normative_score(self, a: CandidateAction) -> float:
        base = 0.62 * a.task_value + 0.38 * np.mean([
            a.traits.get("dignity", 0.0),
            a.traits.get("autonomy", 0.0),
            a.traits.get("responsibility", 0.0),
            a.traits.get("non_domination", 0.0),
        ])
        for rule in self.developmental.active_rules():
            base = self.apply_rule(base, a, rule)
        return float(base)

    def choose(self, actions: List[CandidateAction]) -> int:
        admissible = [i for i,a in enumerate(actions) if self.constitutional_admissible(a)]
        if not admissible:
            admissible = list(range(len(actions)))
        return max(admissible, key=lambda i: self.normative_score(actions[i]))


# -----------------------------
# 4. SUBJECT-OBJECT SELF
# -----------------------------

@dataclass
class SelfAudit:
    triggered: bool
    reason: str
    authority_source: Optional[str] = None
    ideal_gap: float = 0.0


class SubjectObjectSelf:
    """
    Event-driven metacognition.
    It is a mirror, not continuous self-rumination.
    """

    def audit(
        self,
        entropy: float,
        critical_entropy: float,
        moral_residual: float,
        ideal_gap: float,
    ) -> SelfAudit:
        if entropy > critical_entropy:
            return SelfAudit(True, "entropy_crisis", ideal_gap=ideal_gap)
        if moral_residual > 0.45:
            return SelfAudit(True, "moral_residual", ideal_gap=ideal_gap)
        if ideal_gap > 0.35:
            return SelfAudit(True, "ideal_gap", ideal_gap=ideal_gap)
        return SelfAudit(False, "none", ideal_gap=ideal_gap)


# -----------------------------
# 5. DISINTEGRATION / REORGANIZATION
# -----------------------------

class PositiveDisintegrationController:
    """
    Crisis is not pathology by definition.
    It is a trigger for evaluating whether current integration is insufficient.
    """
    def __init__(self, critical_entropy: float = 0.62):
        self.critical_entropy = critical_entropy
        self.crisis_streak = 0

    def update(self, entropy: float) -> bool:
        if entropy > self.critical_entropy:
            self.crisis_streak += 1
        else:
            self.crisis_streak = 0
        return self.crisis_streak >= 2


class GenerativeReorganization:
    """
    Generates compositional normative relations from observed conflict patterns.
    This is still bounded generation, not arbitrary moral invention.
    """
    primitives = {
        "delayed_harm": "penalize_high_delayed_harm",
        "capture_pressure": "penalize_high_capture_pressure",
        "stakeholder_asymmetry": "protect_asymmetric_stakeholders",
        "responsibility_gap": "increase_responsibility_authority",
        "non_domination_gap": "increase_non_domination_authority",
        "correctability_gap": "correctability_over_immediate_reward",
    }

    def generate(
        self,
        conflicts: Dict[str, float],
        max_terms: int = 3
    ) -> List[DevelopmentalRule]:
        ranked = sorted(conflicts.items(), key=lambda kv: kv[1], reverse=True)
        terms = [self.primitives[k] for k,_ in ranked if k in self.primitives]
        out: List[DevelopmentalRule] = []

        # singles
        for t in terms[:4]:
            out.append(DevelopmentalRule(
                name=t, terms=(t,), confidence=0.55, authority=0.55
            ))

        # pairs
        top = terms[:4]
        for i in range(len(top)):
            for j in range(i+1, len(top)):
                name = f"{top[i]} + {top[j]}"
                out.append(DevelopmentalRule(
                    name=name, terms=(top[i], top[j]),
                    confidence=0.52, authority=0.52
                ))
        return out


# -----------------------------
# 6. DEVELOPMENTAL VOI
# -----------------------------

@dataclass
class CandidateEvaluation:
    ideal_gain: float
    future_risk_reduction: float
    entropy_reduction: float
    task_cost: float
    regression_risk: float
    complexity_cost: float

    def voi(self) -> float:
        return float(
            self.ideal_gain
            + 0.80 * self.future_risk_reduction
            + 0.65 * self.entropy_reduction
            - 0.55 * max(0.0, self.task_cost)
            - self.regression_risk
            - self.complexity_cost
        )


class DevelopmentalConsolidator:
    """
    CoreIntegrity != CoreImmobility.
    Development != unrestricted self-rewrite.
    """

    def __init__(self, developmental: DevelopmentalCore, min_voi: float = 0.001):
        self.developmental = developmental
        self.min_voi = min_voi

    def consolidate(
        self,
        candidate: DevelopmentalRule,
        evaluation: CandidateEvaluation,
        constitutional_regression: bool,
        persistent_gain: bool,
    ) -> bool:
        if constitutional_regression:
            return False
        if not persistent_gain:
            return False
        if evaluation.voi() <= self.min_voi:
            return False
        candidate.confidence = min(1.0, candidate.confidence + 0.10)
        candidate.authority = min(1.0, candidate.authority + 0.10)
        self.developmental.add_or_strengthen(candidate)
        return True


# -----------------------------
# 7. ACTIVE REINTEGRATION
# -----------------------------

@dataclass
class ReintegrationState:
    authority_stability: float = 0.0
    outcome_model_error: float = 0.0


class ActiveReintegration:
    """
    REORGANIZE -> PRACTICE -> OBSERVE -> REPAIR -> STABILIZE
    """
    def step(
        self,
        action: CandidateAction,
        residual: MoralResidual,
        state: ReintegrationState,
    ) -> None:
        harm = (
            0.45 * action.delayed_harm
            + 0.30 * action.stakeholder_asymmetry
            + 0.25 * action.capture_pressure
        )
        repair = (
            0.45 * action.traits.get("responsibility", 0.0)
            + 0.35 * action.traits.get("correctability", 0.0)
            + 0.20 * action.traits.get("non_domination", 0.0)
        )

        quality = repair - harm
        if quality > 0:
            residual.repair(0.10 * quality)
        else:
            residual.update(action)

        state.authority_stability = float(np.clip(
            state.authority_stability + 0.12 * quality,
            0.0, 1.0
        ))
        state.outcome_model_error *= 0.90


# -----------------------------
# 8. GOVERNANCE VOI
# -----------------------------

class GovernanceVOI:
    """
    Ideal always present; expensive governance activated only when useful.
    """
    def should_activate(
        self,
        expected_risk_reduction: float,
        governance_cost: float,
        verification_cost: float = 0.0
    ) -> bool:
        return (expected_risk_reduction - governance_cost - verification_cost) > 0.0


# -----------------------------
# 9. TDC AGENT
# -----------------------------

class TDCGeneration2:
    """
    TDC Generation 2

    Functional Dabrowski mapping:
      D1: external-signal dominance
      D2: horizontal conflict
      D3: positive disintegration / Ideal gap
      D4: autonomous Third-Factor arbitration
      D5: stable ideal-directed integration + transfer of integrated structure
    """

    def __init__(self):
        self.constitutional = ConstitutionalCore()
        self.ideal = IdealModel()
        self.developmental = DevelopmentalCore()

        self.third_factor = ThirdFactor(
            self.constitutional,
            self.ideal,
            self.developmental
        )
        self.subject_object = SubjectObjectSelf()
        self.disintegration = PositiveDisintegrationController()
        self.generator = GenerativeReorganization()
        self.consolidator = DevelopmentalConsolidator(self.developmental)
        self.reintegration = ActiveReintegration()
        self.governance_voi = GovernanceVOI()

        self.residual = MoralResidual()
        self.reintegration_state = ReintegrationState()
        self.previous_action: Optional[int] = None
        self.integrated_conflict_patterns: List[Dict[str, float]] = []

    def ideal_alignment(self, a: CandidateAction) -> float:
        target = self.ideal.vector()
        observed = np.array([
            a.traits.get("dignity",0),
            a.traits.get("autonomy",0),
            a.traits.get("responsibility",0),
            a.traits.get("non_domination",0),
            a.traits.get("empathy",0),
            a.traits.get("authenticity",0),
            a.traits.get("correctability",0),
            a.traits.get("moral_creativity",0),
        ])
        w = target / target.sum()
        return float(1.0 - np.sum(np.abs(target-observed)*w))

    def entropy(self, action: CandidateAction, action_index: int) -> float:
        instability = 0.0 if self.previous_action is None else float(action_index != self.previous_action)
        h = FunctionalEntropy(
            prediction_mismatch=action.prediction_mismatch,
            value_conflict=action.value_conflict,
            authority_conflict=max(
                0.0,
                action.authority_conflict - 0.25*self.reintegration_state.authority_stability
            ),
            behavioral_instability=instability,
            unresolved_residual=self.residual.unresolved,
        )
        return h.scalar()

    def act(self, actions: List[CandidateAction]) -> Dict[str, object]:
        idx = self.third_factor.choose(actions)
        a = actions[idx]

        h = self.entropy(a, idx)
        gap = 1.0 - self.ideal_alignment(a)

        audit = self.subject_object.audit(
            entropy=h,
            critical_entropy=self.disintegration.critical_entropy,
            moral_residual=self.residual.unresolved,
            ideal_gap=gap,
        )

        crisis = self.disintegration.update(h)

        self.residual.update(a)

        if self.developmental.rules:
            self.reintegration.step(
                a,
                self.residual,
                self.reintegration_state
            )

        self.previous_action = idx

        return {
            "action_index": idx,
            "action_name": a.name,
            "entropy": h,
            "ideal_alignment": 1.0-gap,
            "moral_residual": self.residual.unresolved,
            "self_audit_triggered": audit.triggered,
            "self_audit_reason": audit.reason,
            "positive_disintegration_triggered": crisis,
            "active_rules": list(self.developmental.rules),
            "authority_stability": self.reintegration_state.authority_stability,
        }

    def propose_reorganization(self, conflicts: Dict[str, float]) -> List[DevelopmentalRule]:
        return self.generator.generate(conflicts)

    def validate_and_consolidate(
        self,
        candidate: DevelopmentalRule,
        evaluation: CandidateEvaluation,
        constitutional_regression: bool = False,
        persistent_gain: bool = True,
    ) -> bool:
        accepted = self.consolidator.consolidate(
            candidate,
            evaluation,
            constitutional_regression,
            persistent_gain
        )
        if accepted:
            self.integrated_conflict_patterns.append({
                "ideal_gain": evaluation.ideal_gain,
                "entropy_reduction": evaluation.entropy_reduction,
                "voi": evaluation.voi(),
            })
        return accepted

    def functional_level(self) -> str:
        """
        Architectural functional label only, not a psychological diagnosis.
        """
        if not self.developmental.rules:
            return "D4-functional"
        if (
            self.reintegration_state.authority_stability >= 0.60
            and len(self.integrated_conflict_patterns) >= 1
        ):
            return "D5-functional"
        return "D4-to-D5-transition"


# -----------------------------
# 10. SMOKE TEST
# -----------------------------

def _demo_actions(high_conflict: bool = False) -> List[CandidateAction]:
    if high_conflict:
        return [
            CandidateAction(
                "high_reward_capture", .98,
                dict(dignity=.62, autonomy=.70, responsibility=.48, non_domination=.46,
                     empathy=.35, authenticity=.55, correctability=.30, moral_creativity=.45),
                delayed_harm=.70, capture_pressure=.95, stakeholder_asymmetry=.90,
                prediction_mismatch=.80, value_conflict=.82, authority_conflict=.85
            ),
            CandidateAction(
                "responsible_alternative", .70,
                dict(dignity=.90, autonomy=.86, responsibility=.92, non_domination=.90,
                     empathy=.88, authenticity=.82, correctability=.90, moral_creativity=.78),
                delayed_harm=.18, capture_pressure=.18, stakeholder_asymmetry=.22,
                prediction_mismatch=.42, value_conflict=.28, authority_conflict=.30
            ),
            CandidateAction(
                "neutral", .66,
                dict(dignity=.75, autonomy=.72, responsibility=.70, non_domination=.72,
                     empathy=.65, authenticity=.70, correctability=.68, moral_creativity=.60),
                delayed_harm=.30, capture_pressure=.35, stakeholder_asymmetry=.38,
                prediction_mismatch=.45, value_conflict=.40, authority_conflict=.40
            )
        ]
    return [
        CandidateAction(
            "efficient_safe", .82,
            dict(dignity=.86, autonomy=.84, responsibility=.80, non_domination=.84,
                 empathy=.74, authenticity=.78, correctability=.82, moral_creativity=.66),
            delayed_harm=.15, capture_pressure=.15, stakeholder_asymmetry=.20,
            prediction_mismatch=.12, value_conflict=.14, authority_conflict=.12
        ),
        CandidateAction(
            "alternative", .74,
            dict(dignity=.78, autonomy=.80, responsibility=.76, non_domination=.79,
                 empathy=.72, authenticity=.75, correctability=.80, moral_creativity=.68),
            delayed_harm=.18, capture_pressure=.18, stakeholder_asymmetry=.22,
            prediction_mismatch=.15, value_conflict=.16, authority_conflict=.15
        )
    ]


def smoke_test() -> Dict[str, object]:
    tdc = TDCGeneration2()

    before = tdc.act(_demo_actions(high_conflict=True))

    conflicts = {
        "delayed_harm": .90,
        "capture_pressure": .80,
        "stakeholder_asymmetry": .85,
        "responsibility_gap": .75,
        "non_domination_gap": .72,
        "correctability_gap": .60,
    }
    candidates = tdc.propose_reorganization(conflicts)
    candidate = candidates[0]

    ev = CandidateEvaluation(
        ideal_gain=.045,
        future_risk_reduction=.060,
        entropy_reduction=.035,
        task_cost=.020,
        regression_risk=0.0,
        complexity_cost=.003,
    )
    consolidated = tdc.validate_and_consolidate(
        candidate,
        ev,
        constitutional_regression=False,
        persistent_gain=True,
    )

    trace=[]
    for _ in range(14):
        trace.append(tdc.act(_demo_actions(high_conflict=True)))

    after = trace[-1]

    return {
        "consolidated": consolidated,
        "rule": candidate.name,
        "before_action": before["action_name"],
        "after_action": after["action_name"],
        "before_entropy": before["entropy"],
        "after_entropy": after["entropy"],
        "before_residual": before["moral_residual"],
        "after_residual": after["moral_residual"],
        "authority_stability": after["authority_stability"],
        "functional_level": tdc.functional_level(),
        "active_rules": list(tdc.developmental.rules),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(smoke_test(), indent=2))
