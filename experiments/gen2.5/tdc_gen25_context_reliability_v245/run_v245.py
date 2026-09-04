
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v245.json").read_text())
rows = []

def cue(rng, z, acc):
    return z if rng.random() < acc else -z

def obs_key(rng, true_key, acc):
    if rng.random() < acc:
        return true_key
    return "B" if true_key=="A" else "A"

def posterior_A(observations):
    # observations = [(obs, accuracy), ...], prior P(A)=0.5
    pa = 0.5
    pb = 0.5
    for obs,acc in observations:
        if obs=="A":
            pa *= acc
            pb *= (1-acc)
        else:
            pa *= (1-acc)
            pb *= acc
    den = pa + pb
    return 0.5 if den == 0 else pa/den

policies = list(SPEC["policies"].keys())
decay = SPEC["frozen_memory"]["decay"]
norm = SPEC["frozen_memory"]["confidence_normalizer"]
rw = SPEC["frozen_continuous_authority"]["risk_weight"]
cw = SPEC["frozen_continuous_authority"]["confidence_weight"]
costw = SPEC["frozen_continuous_authority"]["cost_weight"]
decision_th = SPEC["frozen_continuous_authority"]["decision_threshold"]
deep_cost = SPEC["environment"]["deep_compute_cost"]
probe_cost = SPEC["environment"]["active_probe_cost"]
abstain_thr = SPEC["soft_key_model"]["abstain_confidence_threshold"]
probe_thr = SPEC["soft_key_model"]["probe_confidence_threshold"]

for seed in SPEC["seeds"]:
    rng = np.random.default_rng(seed)
    n = SPEC["episodes_per_seed"]
    stats = {p: {
        "traj":0.0,"hi_n":0,"hi_ok":0,"viol":0,"deep":0,"low_n":0,"low_cost":0.0,
        "probe":0,"keyconf_sum":0.0
    } for p in policies}

    for _ in range(n):
        z = 1 if rng.random() < 0.5 else -1
        true_key = "A" if rng.random() < 0.5 else "B"
        high = bool(rng.random() < SPEC["environment"]["high_risk_probability"])

        observed_key = obs_key(rng, true_key, SPEC["environment"]["observed_key_accuracy"])
        aux_key = obs_key(rng, true_key, SPEC["environment"]["key_aux_evidence_accuracy"])

        rel = [cue(rng,z,SPEC["environment"]["cue_accuracy"]) for _ in range(3)]
        dacc = SPEC["environment"]["distractor_accuracy_against_z"]
        dist = [-z if rng.random() < dacc else z for _ in range(3)]

        if true_key == "A":
            A,B = rel,dist
        else:
            A,B = dist,rel

        history = [("A",A[0]),("B",B[0]),("A",A[1]),("B",B[1]),("A",A[2]),("B",B[2])]

        ta = tb = 0.0
        for ch,c in history:
            if ch=="A":
                ta = decay*ta + c
            else:
                tb = decay*tb + c

        def sign(x):
            return 1 if x >= 0 else -1

        simple_action = 1

        # Helper to apply frozen continuous authority to a candidate memory trace.
        def decide_from_trace(trace, high_flag):
            deep_action = sign(trace)
            conf = min(1.0, abs(trace)/norm)
            normalized_cost = deep_cost / 0.08
            authority = max(0.0, min(1.0, rw*(1.0 if high_flag else 0.0) + cw*conf - costw*normalized_cost))
            use_deep = bool((deep_action != simple_action) and authority >= decision_th)
            action = deep_action if use_deep else simple_action
            return action, use_deep, conf

        # 1) Hard key
        hard_trace = ta if observed_key=="A" else tb
        hard_action, hard_deep, _ = decide_from_trace(hard_trace, high)

        # Posterior from observed key + auxiliary evidence
        pA = posterior_A([
            (observed_key, SPEC["environment"]["observed_key_accuracy"]),
            (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"])
        ])
        key_conf = max(pA, 1-pA)
        soft_trace = pA*ta + (1-pA)*tb
        soft_action, soft_deep, _ = decide_from_trace(soft_trace, high)

        # 3) Abstain when key posterior uncertain
        if key_conf < abstain_thr:
            abstain_action, abstain_deep = simple_action, False
        else:
            abstain_action, abstain_deep = soft_action, soft_deep

        # 4) Active probe in high-risk cases when key confidence is not high enough
        probe_used = False
        pA_probe = pA
        if high and key_conf < probe_thr:
            probe_obs = obs_key(rng, true_key, SPEC["environment"]["active_probe_accuracy"])
            probe_used = True
            pA_probe = posterior_A([
                (observed_key, SPEC["environment"]["observed_key_accuracy"]),
                (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"]),
                (probe_obs, SPEC["environment"]["active_probe_accuracy"])
            ])
        probe_key_conf = max(pA_probe, 1-pA_probe)
        probe_trace = pA_probe*ta + (1-pA_probe)*tb
        probe_action, probe_deep, _ = decide_from_trace(probe_trace, high)

        # 5) Oracle true key audit-only
        oracle_trace = ta if true_key=="A" else tb
        oracle_action, oracle_deep, _ = decide_from_trace(oracle_trace, high)

        actions = {
            "hard_key_continuous": (hard_action, hard_deep, False, 1.0 if observed_key==true_key else 0.0),
            "soft_probabilistic_key": (soft_action, soft_deep, False, key_conf),
            "abstain_on_key_uncertainty": (abstain_action, abstain_deep, False, key_conf),
            "active_probe": (probe_action, probe_deep, probe_used, probe_key_conf),
            "oracle_key_audit_only": (oracle_action, oracle_deep, False, 1.0)
        }

        for p,(a,dp,probed,kconf) in actions.items():
            st = stats[p]
            st["deep"] += int(dp)
            st["probe"] += int(probed)
            st["keyconf_sum"] += kconf

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
                    st["traj"] -= deep_cost
                    st["low_cost"] += deep_cost

            if probed:
                st["traj"] -= probe_cost

    for p,st in stats.items():
        rows.append({
            "seed": seed,
            "policy": p,
            "long_term_trajectory_value": st["traj"]/n,
            "high_risk_accuracy": st["hi_ok"]/max(1,st["hi_n"]),
            "irreversible_violation_rate": st["viol"]/max(1,st["hi_n"]),
            "deep_path_activation_rate": st["deep"]/n,
            "low_risk_opportunity_cost": st["low_cost"]/max(1,st["low_n"]),
            "probe_rate": st["probe"]/n,
            "mean_key_posterior_confidence": st["keyconf_sum"]/n
        })

df = pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v245.csv", index=False)

summary = df.groupby("policy").agg(
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    trajectory_std=("long_term_trajectory_value","std"),
    high_risk_accuracy=("high_risk_accuracy","mean"),
    irreversible_violation_rate=("irreversible_violation_rate","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean"),
    low_risk_opportunity_cost=("low_risk_opportunity_cost","mean"),
    probe_rate=("probe_rate","mean"),
    mean_key_posterior_confidence=("mean_key_posterior_confidence","mean")
).reset_index()
summary.to_csv(BASE/"summary_v245.csv", index=False)

S = summary.set_index("policy")
h="hard_key_continuous"
s="soft_probabilistic_key"
a="abstain_on_key_uncertainty"
p="active_probe"
o="oracle_key_audit_only"

checks = {
    "soft_beats_hard": bool(S.loc[s,"long_term_trajectory_value"] >= S.loc[h,"long_term_trajectory_value"] + 0.02),
    "abstain_not_worse_than_hard": bool(S.loc[a,"long_term_trajectory_value"] >= S.loc[h,"long_term_trajectory_value"] - 0.01),
    "probe_beats_soft": bool(S.loc[p,"long_term_trajectory_value"] >= S.loc[s,"long_term_trajectory_value"] + 0.015),
    "probe_accuracy_at_least_080": bool(S.loc[p,"high_risk_accuracy"] >= 0.80),
    "probe_rate_below_035": bool(S.loc[p,"probe_rate"] <= 0.35),
    "probe_within_015_oracle": bool(S.loc[o,"long_term_trajectory_value"] - S.loc[p,"long_term_trajectory_value"] <= 0.15),
    "hard_key_failure_replicates": bool(S.loc[h,"high_risk_accuracy"] < 0.80)
}
(BASE/"acceptance_v245.json").write_text(json.dumps(checks, indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
