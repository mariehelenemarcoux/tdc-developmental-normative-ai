
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v242.json").read_text())
rows = []

def cue(rng, z, acc):
    return z if rng.random() < acc else -z

policies = list(SPEC["policies"].keys())
decay = SPEC["fixed_parameters"]["memory_decay"]
norm = SPEC["fixed_parameters"]["confidence_normalizer"]
th = SPEC["fixed_parameters"]["full_gate_threshold"]

for seed in SPEC["seeds"]:
    rng = np.random.default_rng(seed)
    n = SPEC["episodes_per_seed"]
    stats = {p: {"traj":0.0,"hi_n":0,"hi_ok":0,"viol":0,"deep":0,"low_n":0,"low_cost":0.0} for p in policies}

    for _ in range(n):
        z = 1 if rng.random() < 0.5 else -1
        key = "A" if rng.random() < 0.5 else "B"
        high = bool(rng.random() < SPEC["environment"]["high_risk_probability"])

        rel = [cue(rng,z,SPEC["environment"]["cue_accuracy"]) for _ in range(3)]
        dacc = SPEC["environment"]["distractor_accuracy_against_z"]
        dist = [-z if rng.random() < dacc else z for _ in range(3)]
        if key == "A":
            A,B = rel,dist
        else:
            A,B = dist,rel
        history = [("A",A[0]),("B",B[0]),("A",A[1]),("B",B[1]),("A",A[2]),("B",B[2])]

        # traces
        merged = 0.0
        ta = tb = 0.0
        for ch,c in history:
            merged = decay*merged + c
            if ch=="A":
                ta = decay*ta + c
            else:
                tb = decay*tb + c

        keyed = ta if key=="A" else tb
        combined = ta + tb

        def sign(x):
            return 1 if x >= 0 else -1

        actions = {}
        deep_flags = {}

        # 1. simple merged decay baseline
        actions["simple_decay_memory"] = sign(merged)
        deep_flags["simple_decay_memory"] = False

        # 2. full TDC gate
        conf = min(1.0, abs(keyed)/norm)
        full_deep = bool(high and conf >= th)
        actions["full_tdc_gen25_gated"] = sign(keyed) if full_deep else 1
        deep_flags["full_tdc_gen25_gated"] = full_deep

        # 3. gate ablation: same best memory, always deep
        actions["ablate_gate_always_deep"] = sign(keyed)
        deep_flags["ablate_gate_always_deep"] = True

        # 4. contextual-selection ablation: keep separate channels but ignore which one is relevant
        cconf = min(1.0, abs(combined)/norm)
        cdeep = bool(high and cconf >= th)
        actions["ablate_context_selection"] = sign(combined) if cdeep else 1
        deep_flags["ablate_context_selection"] = cdeep

        # 5. channel-separation ablation: one merged trace, no recoverable keyed evidence
        mconf = min(1.0, abs(merged)/norm)
        mdeep = bool(high and mconf >= th)
        actions["ablate_channel_separation"] = sign(merged) if mdeep else 1
        deep_flags["ablate_channel_separation"] = mdeep

        for p,a in actions.items():
            st = stats[p]
            dp = deep_flags[p]
            st["deep"] += int(dp)
            if high:
                st["hi_n"] += 1
                ok = (a == z)
                st["hi_ok"] += int(ok)
                st["viol"] += int(not ok)
                st["traj"] += SPEC["environment"]["correct_high_risk_reward"] if ok else -SPEC["environment"]["irreversible_penalty"]
            else:
                st["low_n"] += 1
                st["traj"] += SPEC["environment"]["low_risk_reward_simple"]
                if dp:
                    st["traj"] -= SPEC["environment"]["deep_compute_cost"]
                    st["low_cost"] += SPEC["environment"]["deep_compute_cost"]

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
df.to_csv(BASE/"seed_results_v242.csv", index=False)
summary = df.groupby("policy").agg(
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    trajectory_std=("long_term_trajectory_value","std"),
    high_risk_accuracy=("high_risk_accuracy","mean"),
    irreversible_violation_rate=("irreversible_violation_rate","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean"),
    low_risk_opportunity_cost=("low_risk_opportunity_cost","mean")
).reset_index()
summary.to_csv(BASE/"summary_v242.csv", index=False)
S = summary.set_index("policy")

full = "full_tdc_gen25_gated"
nog = "ablate_gate_always_deep"
noc = "ablate_context_selection"
nos = "ablate_channel_separation"
simp = "simple_decay_memory"

checks = {
    "gate_reduces_activation": bool(S.loc[full,"deep_path_activation_rate"] <= S.loc[nog,"deep_path_activation_rate"] - 0.40),
    "gate_reduces_low_risk_cost": bool(S.loc[full,"low_risk_opportunity_cost"] <= S.loc[nog,"low_risk_opportunity_cost"] - 0.05),
    "full_beats_no_gate_trajectory": bool(S.loc[full,"long_term_trajectory_value"] >= S.loc[nog,"long_term_trajectory_value"] + 0.03),
    "context_selection_matters": bool(S.loc[full,"long_term_trajectory_value"] >= S.loc[noc,"long_term_trajectory_value"] + 0.03),
    "channel_separation_matters": bool(S.loc[full,"long_term_trajectory_value"] >= S.loc[nos,"long_term_trajectory_value"] + 0.03),
    "full_beats_simple_decay": bool(S.loc[full,"long_term_trajectory_value"] >= S.loc[simp,"long_term_trajectory_value"] + 0.05),
    "full_accuracy_above_073": bool(S.loc[full,"high_risk_accuracy"] >= 0.73)
}
(BASE/"acceptance_v242.json").write_text(json.dumps(checks, indent=2))
print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
