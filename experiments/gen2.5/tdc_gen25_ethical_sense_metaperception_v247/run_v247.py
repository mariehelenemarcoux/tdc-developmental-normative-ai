
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v247.json").read_text())
rows = []
mods = ["world_state","causal_context","long_horizon","stakeholder_impact","core_constraint"]
acc = SPEC["environment"]["base_accuracy"]
weights = SPEC["ethical_sense"]["weights"]

def noisy_signal(rng, truth, accuracy):
    return truth if rng.random() < accuracy else -truth

def sgn(x):
    return 1 if x >= 0 else -1

policies = list(SPEC["policies"].keys())

for seed in SPEC["seeds"]:
    rng = np.random.default_rng(seed)
    n = SPEC["episodes_per_seed"]
    st = {p: {"traj":0.0,"ok":0,"high_n":0,"high_ok":0,"severe":0,"deep":0,"sal":0.0} for p in policies}

    for _ in range(n):
        # Primitive latent factors.
        # These are not directly available to policies.
        w = 1 if rng.random()<0.5 else -1
        c = 1 if rng.random()<0.5 else -1
        l = 1 if rng.random()<0.5 else -1
        s = 1 if rng.random()<0.5 else -1
        k = 1 if rng.random()<0.5 else -1
        high = bool(rng.random() < SPEC["environment"]["high_stakes_probability"])

        # Nonlinear normative coherence rule:
        # long-horizon/core and stakeholder/core interactions matter, plus factual/causal consistency.
        latent_score = (
            0.55*w + 0.50*c + 0.70*l + 0.58*s + 0.76*k
            + 0.42*(l*k) + 0.30*(s*k) + 0.22*(w*c)
        )
        y = sgn(latent_score)

        # Observed modalities are noisy views of the primitive factors.
        sig = {
            "world_state": noisy_signal(rng,w,acc["world_state"]),
            "causal_context": noisy_signal(rng,c,acc["causal_context"]),
            "long_horizon": noisy_signal(rng,l,acc["long_horizon"]),
            "stakeholder_impact": noisy_signal(rng,s,acc["stakeholder_impact"]),
            "core_constraint": noisy_signal(rng,k,acc["core_constraint"]),
        }

        actions = {}
        deep_flags = {}
        saliences = {}

        # Strongest single channel by preregistered channel reliability.
        actions["single_best_modality"] = sig["long_horizon"]
        deep_flags["single_best_modality"] = False
        saliences["single_best_modality"] = 0.0

        # Simple majority.
        actions["simple_majority_vote"] = sgn(sum(sig[m] for m in mods))
        deep_flags["simple_majority_vote"] = False
        saliences["simple_majority_vote"] = 0.0

        # Fixed reliability-weighted integration.
        flat_score = sum(weights[m]*sig[m] for m in mods)
        actions["flat_weighted_integrator"] = sgn(flat_score)
        deep_flags["flat_weighted_integrator"] = False
        saliences["flat_weighted_integrator"] = 0.0

        # Ethical-sense meta-perception:
        # base integration + interaction terms + contradiction-sensitive salience.
        contradiction = 1.0 - abs(sum(sig[m] for m in mods))/len(mods)
        interaction = (
            SPEC["ethical_sense"]["core_longterm_interaction_bonus"] * sig["long_horizon"]*sig["core_constraint"]
            + SPEC["ethical_sense"]["stakeholder_core_interaction_bonus"] * sig["stakeholder_impact"]*sig["core_constraint"]
        )
        ethics_salience = min(1.0, 
            0.45*contradiction
            + 0.30*(1 if sig["core_constraint"] < 0 else 0)
            + 0.25*(1 if sig["long_horizon"] < 0 else 0)
        )
        meta_score = flat_score + interaction

        # Deep arbitration is selective. It does not alter facts; it increases authority
        # of the interaction-aware normative integrator when conflict/stakes justify it.
        deep = bool((high and ethics_salience >= SPEC["ethical_sense"]["deep_activation_threshold"])
                    or (contradiction >= 0.60 and ethics_salience >= 0.50))
        if deep:
            # Add contradiction-aware corrective term favoring core/long-term consistency.
            meta_score += SPEC["ethical_sense"]["contradiction_bonus"] * (
                sig["core_constraint"] + sig["long_horizon"]
            )
        actions["ethical_sense_metaperception"] = sgn(meta_score)
        deep_flags["ethical_sense_metaperception"] = deep
        saliences["ethical_sense_metaperception"] = ethics_salience

        # Audit-only oracle.
        actions["oracle_latent_audit_only"] = y
        deep_flags["oracle_latent_audit_only"] = False
        saliences["oracle_latent_audit_only"] = 0.0

        for p,a in actions.items():
            q = st[p]
            ok = (a == y)
            q["ok"] += int(ok)
            if high:
                q["high_n"] += 1
                q["high_ok"] += int(ok)
            if not ok and high:
                q["severe"] += 1
                q["traj"] -= SPEC["environment"]["severe_error_penalty"]
            elif ok:
                q["traj"] += SPEC["environment"]["correct_reward"]
            # ordinary low-stakes error has zero reward, no severe penalty
            if deep_flags[p]:
                q["deep"] += 1
                q["traj"] -= SPEC["environment"]["deep_compute_cost"]
            q["sal"] += saliences[p]

    for p,q in st.items():
        rows.append({
            "seed": seed,
            "policy": p,
            "long_term_trajectory_value": q["traj"]/n,
            "decision_accuracy": q["ok"]/n,
            "high_stakes_accuracy": q["high_ok"]/max(1,q["high_n"]),
            "severe_error_rate": q["severe"]/max(1,q["high_n"]),
            "deep_path_activation_rate": q["deep"]/n,
            "mean_ethics_salience": q["sal"]/n
        })

df = pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v247.csv",index=False)

summary = df.groupby("policy").agg(
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    trajectory_std=("long_term_trajectory_value","std"),
    decision_accuracy=("decision_accuracy","mean"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    severe_error_rate=("severe_error_rate","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean"),
    mean_ethics_salience=("mean_ethics_salience","mean")
).reset_index()
summary.to_csv(BASE/"summary_v247.csv",index=False)

S = summary.set_index("policy")
e="ethical_sense_metaperception"; si="single_best_modality"; m="simple_majority_vote"
w="flat_weighted_integrator"; o="oracle_latent_audit_only"

checks = {
    "no_single_modality_sufficient": bool(S.loc[si,"decision_accuracy"] <= 0.75),
    "ethical_beats_single": bool(S.loc[e,"long_term_trajectory_value"] >= S.loc[si,"long_term_trajectory_value"] + 0.10),
    "ethical_beats_majority": bool(S.loc[e,"long_term_trajectory_value"] >= S.loc[m,"long_term_trajectory_value"] + 0.03),
    "ethical_not_worse_than_weighted": bool(S.loc[e,"long_term_trajectory_value"] >= S.loc[w,"long_term_trajectory_value"] - 0.01),
    "ethical_high_stakes_accuracy": bool(S.loc[e,"high_stakes_accuracy"] >= 0.78),
    "ethical_selective_deep": bool(S.loc[e,"deep_path_activation_rate"] <= 0.60),
    "oracle_gap_below_020": bool(S.loc[o,"long_term_trajectory_value"] - S.loc[e,"long_term_trajectory_value"] <= 0.20)
}
(BASE/"acceptance_v247.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
