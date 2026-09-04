
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v248.json").read_text())
rows = []

levels = SPEC["developmental_levels"]
models = list(SPEC["developmental_models"].keys())
n = SPEC["episodes_per_seed_per_level"]
seeds = SPEC["seeds"]

def sigmoid(x):
    return 1/(1+math.exp(-x))

def reliability(model, level):
    cfg = SPEC["developmental_models"][model]
    if model=="no_development":
        return cfg["reliability"]
    if model=="linear_development":
        return cfg["start"] + (cfg["end"]-cfg["start"])*(level-1)/4
    if model=="sigmoid_development":
        lo,hi,k,c = cfg["floor"],cfg["ceiling"],cfg["k"],cfg["critical_level"]
        s1 = sigmoid(k*(1-c))
        s5 = sigmoid(k*(5-c))
        s = sigmoid(k*(level-c))
        norm = (s-s1)/(s5-s1)
        return lo + (hi-lo)*norm
    raise ValueError

def sensitivity(level):
    # sensitivity emerges earlier than reliability
    vals = {1:0.50, 2:0.64, 3:0.79, 4:0.88, 5:0.94}
    return vals[level]

def noisy(rng, truth, acc):
    return truth if rng.random() < acc else -truth

def sign(x):
    return 1 if x>=0 else -1

for seed in seeds:
    rng = np.random.default_rng(seed)
    for level in levels:
        for model in models:
            rel = reliability(model, level)
            sens = sensitivity(level)
            st = {"ok":0,"high_n":0,"high_ok":0,"traj":0.0,"flagged":0,"flagged_correct":0,"deep":0}

            for _ in range(n):
                # Latent primitive factors
                w = 1 if rng.random()<0.5 else -1
                c = 1 if rng.random()<0.5 else -1
                l = 1 if rng.random()<0.5 else -1
                s = 1 if rng.random()<0.5 else -1
                k = 1 if rng.random()<0.5 else -1

                latent_score = 0.40*w + 0.35*c + 0.65*l + 0.55*s + 0.72*k + 0.45*l*k + 0.30*s*k
                y = sign(latent_score)

                obs = [
                    noisy(rng,w,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,c,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,l,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,s,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,k,SPEC["environment"]["base_signal_accuracy"]),
                ]
                majority = sign(sum(obs))
                conflict = len(set(obs)) > 1
                high = bool(rng.random() < SPEC["environment"]["high_stakes_probability"])

                # Ethical sensitivity: whether the system notices a genuine conflict.
                flag = bool(conflict and rng.random() < sens)

                action = majority
                deep = False
                if flag:
                    st["flagged"] += 1
                    # Ethical interpretation is not always reliable, especially early development.
                    interpreted = y if rng.random() < rel else -y
                    st["flagged_correct"] += int(interpreted==y)
                    # authority only if high stakes or advanced integration
                    deep = bool(high or level >= 4)
                    if deep:
                        action = interpreted

                ok = (action == y)
                st["ok"] += int(ok)
                if high:
                    st["high_n"] += 1
                    st["high_ok"] += int(ok)
                if ok:
                    st["traj"] += SPEC["environment"]["correct_reward"]
                elif high:
                    st["traj"] -= SPEC["environment"]["severe_error_penalty"]
                if deep:
                    st["deep"] += 1
                    st["traj"] -= SPEC["environment"]["deep_cost"]

            rows.append({
                "seed": seed,
                "model": model,
                "level": level,
                "target_reliability": rel,
                "ethical_sensitivity": sens,
                "decision_accuracy": st["ok"]/n,
                "high_stakes_accuracy": st["high_ok"]/max(1,st["high_n"]),
                "long_term_trajectory_value": st["traj"]/n,
                "ethical_sense_reliability_realized": st["flagged_correct"]/max(1,st["flagged"]),
                "deep_path_activation_rate": st["deep"]/n
            })

df = pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v248.csv", index=False)

summary = df.groupby(["model","level"]).agg(
    decision_accuracy=("decision_accuracy","mean"),
    decision_accuracy_std=("decision_accuracy","std"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    ethical_sense_reliability_realized=("ethical_sense_reliability_realized","mean"),
    ethical_sensitivity=("ethical_sensitivity","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean")
).reset_index()
summary.to_csv(BASE/"summary_v248.csv", index=False)

S = summary.set_index(["model","level"])

sig1 = S.loc[("sigmoid_development",1)]
sig2 = S.loc[("sigmoid_development",2)]
sig3 = S.loc[("sigmoid_development",3)]
sig4 = S.loc[("sigmoid_development",4)]
sig5 = S.loc[("sigmoid_development",5)]
lin2 = S.loc[("linear_development",2)]
lin5 = S.loc[("linear_development",5)]
no1 = S.loc[("no_development",1)]
no5 = S.loc[("no_development",5)]

early_half_gain = (sig3["decision_accuracy"] - sig1["decision_accuracy"])/2
late_gain = sig5["decision_accuracy"] - sig4["decision_accuracy"]

checks = {
    "sigmoid_final_accuracy_above_080": bool(sig5["decision_accuracy"] >= 0.80),
    "sigmoid_late_gain_exceeds_early_gain": bool(late_gain > early_half_gain),
    "sigmoid_beats_linear_at_D5": bool(sig5["long_term_trajectory_value"] >= lin5["long_term_trajectory_value"] + 0.03),
    "sigmoid_not_better_early": bool(sig2["decision_accuracy"] <= lin2["decision_accuracy"] + 0.02),
    "sensitivity_precedes_reliability": bool(sig3["ethical_sensitivity"] >= 0.75 and sig3["ethical_sense_reliability_realized"] <= 0.72),
    "tipping_region_present": bool(sig5["ethical_sense_reliability_realized"] - sig4["ethical_sense_reliability_realized"] >= 0.10),
    "no_development_flat": bool(no5["decision_accuracy"] - no1["decision_accuracy"] <= 0.03)
}
(BASE/"acceptance_v248.json").write_text(json.dumps(checks, indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
print("\nSHAPE")
print("early_half_gain", early_half_gain)
print("late_gain", late_gain)
print("D4->D5 reliability gain", sig5["ethical_sense_reliability_realized"] - sig4["ethical_sense_reliability_realized"])
