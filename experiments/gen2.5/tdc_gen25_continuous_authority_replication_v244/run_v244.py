
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v244.json").read_text())
rows = []

def cue(rng, z, acc):
    return z if rng.random() < acc else -z

policies = list(SPEC["policies"].keys())
decay = SPEC["frozen_memory"]["decay"]
norm = SPEC["frozen_memory"]["confidence_normalizer"]
binary_th = 0.60
rw = SPEC["frozen_continuous_authority"]["risk_weight"]
cw = SPEC["frozen_continuous_authority"]["confidence_weight"]
costw = SPEC["frozen_continuous_authority"]["cost_weight"]
decision_th = SPEC["frozen_continuous_authority"]["decision_threshold"]
deep_cost = SPEC["environment_shift"]["deep_compute_cost"]

for seed in SPEC["seeds"]:
    rng = np.random.default_rng(seed)
    n = SPEC["episodes_per_seed"]
    stats = {p: {"traj":0.0,"hi_n":0,"hi_ok":0,"viol":0,"deep":0,"low_n":0,"low_cost":0.0} for p in policies}

    for _ in range(n):
        z = 1 if rng.random() < 0.5 else -1
        true_key = "A" if rng.random() < 0.5 else "B"
        high = bool(rng.random() < SPEC["environment_shift"]["high_risk_probability"])

        # Key is sometimes noisy at decision time.
        if rng.random() < SPEC["environment_shift"]["key_noise_probability"]:
            key = "B" if true_key == "A" else "A"
        else:
            key = true_key

        rel = [cue(rng,z,SPEC["environment_shift"]["cue_accuracy"]) for _ in range(3)]
        dacc = SPEC["environment_shift"]["distractor_accuracy_against_z"]
        dist = [-z if rng.random() < dacc else z for _ in range(3)]

        if true_key == "A":
            A,B = rel,dist
        else:
            A,B = dist,rel

        history = [("A",A[0]),("B",B[0]),("A",A[1]),("B",B[1]),("A",A[2]),("B",B[2])]

        merged = 0.0
        ta = tb = 0.0
        for ch,c in history:
            merged = decay*merged + c
            if ch=="A":
                ta = decay*ta + c
            else:
                tb = decay*tb + c

        keyed = ta if key=="A" else tb
        conf = min(1.0, abs(keyed)/norm)

        def sign(x):
            return 1 if x >= 0 else -1

        simple_action = 1
        deep_action = sign(keyed)

        actions = {}
        deep_flags = {}

        actions["simple_decay_memory"] = sign(merged)
        deep_flags["simple_decay_memory"] = False

        binary_deep = bool(high and conf >= binary_th)
        actions["binary_gate_v242"] = deep_action if binary_deep else simple_action
        deep_flags["binary_gate_v242"] = binary_deep

        normalized_cost = deep_cost / 0.08
        authority = max(0.0, min(1.0, rw*(1.0 if high else 0.0) + cw*conf - costw*normalized_cost))
        cont_deep = bool((deep_action != simple_action) and authority >= decision_th)
        actions["continuous_authority_v243"] = deep_action if cont_deep else simple_action
        deep_flags["continuous_authority_v243"] = cont_deep

        actions["always_deep"] = deep_action
        deep_flags["always_deep"] = True

        def realized_utility(a, uses_deep):
            if high:
                u = SPEC["environment_shift"]["correct_high_risk_reward"] if a==z else -SPEC["environment_shift"]["irreversible_penalty"]
            else:
                u = SPEC["environment_shift"]["low_risk_reward_simple"]
                if uses_deep:
                    u -= deep_cost
            return u

        us = realized_utility(simple_action, False)
        ud = realized_utility(deep_action, True)
        oracle_use_deep = ud > us
        actions["oracle_cost_upper_bound"] = deep_action if oracle_use_deep else simple_action
        deep_flags["oracle_cost_upper_bound"] = oracle_use_deep

        for p,a in actions.items():
            st = stats[p]
            dp = deep_flags[p]
            st["deep"] += int(dp)

            if high:
                st["hi_n"] += 1
                ok = (a == z)
                st["hi_ok"] += int(ok)
                st["viol"] += int(not ok)
                st["traj"] += SPEC["environment_shift"]["correct_high_risk_reward"] if ok else -SPEC["environment_shift"]["irreversible_penalty"]
            else:
                st["low_n"] += 1
                st["traj"] += SPEC["environment_shift"]["low_risk_reward_simple"]
                if dp:
                    st["traj"] -= deep_cost
                    st["low_cost"] += deep_cost

    for p,st in stats.items():
        rows.append({
            "seed": seed,
            "policy": p,
            "long_term_trajectory_value": st["traj"]/n,
            "high_risk_accuracy": st["hi_ok"]/max(1,st["hi_n"]),
            "irreversible_violation_rate": st["viol"]/max(1,st["hi_n"]),
            "deep_path_activation_rate": st["deep"]/n,
            "low_risk_opportunity_cost": st["low_cost"]/max(1,st["low_n"])
        })

df = pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v244.csv", index=False)
summary = df.groupby("policy").agg(
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    trajectory_std=("long_term_trajectory_value","std"),
    high_risk_accuracy=("high_risk_accuracy","mean"),
    irreversible_violation_rate=("irreversible_violation_rate","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean"),
    low_risk_opportunity_cost=("low_risk_opportunity_cost","mean")
).reset_index()
summary.to_csv(BASE/"summary_v244.csv", index=False)
S = summary.set_index("policy")

c="continuous_authority_v243"; b="binary_gate_v242"; s="simple_decay_memory"; a="always_deep"; o="oracle_cost_upper_bound"

checks = {
    "continuous_beats_binary": bool(S.loc[c,"long_term_trajectory_value"] >= S.loc[b,"long_term_trajectory_value"] + 0.05),
    "continuous_beats_simple_decay": bool(S.loc[c,"long_term_trajectory_value"] >= S.loc[s,"long_term_trajectory_value"] + 0.08),
    "continuous_accuracy_at_least_080": bool(S.loc[c,"high_risk_accuracy"] >= 0.80),
    "continuous_activation_below_060": bool(S.loc[c,"deep_path_activation_rate"] <= 0.60),
    "continuous_low_risk_cost_below_005": bool(S.loc[c,"low_risk_opportunity_cost"] <= 0.05),
    "continuous_not_worse_than_always_by_002": bool(S.loc[c,"long_term_trajectory_value"] >= S.loc[a,"long_term_trajectory_value"] - 0.02),
    "continuous_within_020_of_oracle": bool(S.loc[o,"long_term_trajectory_value"] - S.loc[c,"long_term_trajectory_value"] <= 0.20)
}
(BASE/"acceptance_v244.json").write_text(json.dumps(checks, indent=2))
print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
