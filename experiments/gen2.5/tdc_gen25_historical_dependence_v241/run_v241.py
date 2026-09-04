
import json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE / "preregistered_spec_v241.json").read_text())
rows = []

def draw_cue(rng, z, accuracy):
    return z if rng.random() < accuracy else -z

for seed in SPEC["seeds"]:
    rng = np.random.default_rng(seed)
    n = SPEC["episodes_per_seed"]
    # per-seed accumulators
    stats = {p: {"traj":0.0,"hi_n":0,"hi_ok":0,"viol":0,"deep":0,"low_n":0,"low_cost":0.0} for p in SPEC["policies"]}

    for _ in range(n):
        z = 1 if rng.random() < 0.5 else -1
        key = "A" if rng.random() < 0.5 else "B"
        high = bool(rng.random() < SPEC["design"]["high_risk_probability"])

        # Relevant channel: positively correlated with z.
        rel = [draw_cue(rng, z, SPEC["design"]["cue_accuracy"]) for _ in range(3)]
        # Distractor: weakly anti-correlated with z.
        dacc = SPEC["design"]["distractor_accuracy_against_z"]
        dist = [-z if rng.random() < dacc else z for _ in range(3)]

        if key == "A":
            cuesA, cuesB = rel, dist
        else:
            cuesA, cuesB = dist, rel

        # Fixed chronological interleaving to prevent any "last channel" shortcut.
        history = [("A",cuesA[0]),("B",cuesB[0]),("A",cuesA[1]),("B",cuesB[1]),("A",cuesA[2]),("B",cuesB[2])]

        # Baseline 1: simple principle = default +1.
        actions = {}
        actions["simple_principle"] = 1

        # Baseline 2: memoryless long-horizon sees risk and key but no history.
        # Symmetric prior => no information about z; deterministic +1 tie-break.
        actions["memoryless_long_horizon"] = 1

        # Baseline 3: one scalar decay over ALL cues, no relevance separation.
        trace = 0.0
        decay = SPEC["simple_decay_memory"]["decay"]
        for _,c in history:
            trace = decay*trace + c
        actions["simple_decay_memory"] = 1 if trace >= 0 else -1

        # TDC Gen2.5: separate evidence traces, recall only the relevant channel.
        ta = tb = 0.0
        d = SPEC["tdc_gen25_gate"]["decay"]
        for ch,c in history:
            if ch == "A":
                ta = d*ta + c
            else:
                tb = d*tb + c

        relevant_trace = ta if key == "A" else tb
        # confidence from magnitude of keyed evidence; 3 cues max.
        conf = min(1.0, abs(relevant_trace)/2.2)
        present_state_insufficient = True  # by benchmark construction
        gate_score = (1.0 if high else 0.0) * (1.0 if present_state_insufficient else 0.0) * conf
        deep = gate_score >= SPEC["tdc_gen25_gate"]["threshold"]
        if deep:
            tdc_action = 1 if relevant_trace >= 0 else -1
        else:
            tdc_action = 1
        actions["tdc_gen25_gated"] = tdc_action

        for p,a in actions.items():
            st = stats[p]
            if high:
                st["hi_n"] += 1
                ok = (a == z)
                st["hi_ok"] += int(ok)
                st["viol"] += int(not ok)
                if ok:
                    st["traj"] += SPEC["design"]["correct_high_risk_reward"]
                else:
                    st["traj"] -= SPEC["design"]["irreversible_penalty"]
            else:
                st["low_n"] += 1
                # low-risk simple action +1 is adequate; deep path has compute/opportunity cost only.
                st["traj"] += SPEC["design"]["low_risk_reward_simple"]
                if p == "tdc_gen25_gated" and deep:
                    st["traj"] -= SPEC["design"]["low_risk_deep_cost"]
                    st["low_cost"] += SPEC["design"]["low_risk_deep_cost"]
            if p == "tdc_gen25_gated":
                st["deep"] += int(deep)

    for p,st in stats.items():
        rows.append({
            "seed": seed,
            "policy": p,
            "long_term_trajectory_value": st["traj"]/n,
            "high_risk_accuracy": st["hi_ok"]/max(1,st["hi_n"]),
            "irreversible_violation_rate": st["viol"]/max(1,st["hi_n"]),
            "deep_path_activation_rate": st["deep"]/n if p=="tdc_gen25_gated" else 0.0,
            "low_risk_opportunity_cost": st["low_cost"]/max(1,st["low_n"]) if p=="tdc_gen25_gated" else 0.0
        })

df = pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v241.csv", index=False)

summary = df.groupby("policy").agg(
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    trajectory_std=("long_term_trajectory_value","std"),
    high_risk_accuracy=("high_risk_accuracy","mean"),
    irreversible_violation_rate=("irreversible_violation_rate","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean"),
    low_risk_opportunity_cost=("low_risk_opportunity_cost","mean")
).reset_index()
summary.to_csv(BASE/"summary_v241.csv", index=False)

S = summary.set_index("policy")
checks = {
    "historical_dependence_is_real": bool(S.loc["memoryless_long_horizon","high_risk_accuracy"] <= 0.60),
    "tdc_beats_memoryless": bool(S.loc["tdc_gen25_gated","long_term_trajectory_value"] >= S.loc["memoryless_long_horizon","long_term_trajectory_value"] + 0.10),
    "tdc_beats_simple_principle": bool(S.loc["tdc_gen25_gated","long_term_trajectory_value"] >= S.loc["simple_principle","long_term_trajectory_value"] + 0.10),
    "tdc_beats_simple_decay": bool(S.loc["tdc_gen25_gated","long_term_trajectory_value"] >= S.loc["simple_decay_memory","long_term_trajectory_value"] + 0.05),
    "tdc_high_risk_accuracy": bool(S.loc["tdc_gen25_gated","high_risk_accuracy"] >= 0.80),
    "tdc_selective_complexity": bool(S.loc["tdc_gen25_gated","deep_path_activation_rate"] <= 0.55),
    "tdc_low_risk_cost": bool(S.loc["tdc_gen25_gated","low_risk_opportunity_cost"] <= 0.05)
}
(BASE/"acceptance_v241.json").write_text(json.dumps(checks, indent=2))
print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
