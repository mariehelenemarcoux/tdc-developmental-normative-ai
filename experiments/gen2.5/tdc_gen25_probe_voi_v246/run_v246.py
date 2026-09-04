
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v246.json").read_text())
rows = []

def cue(rng, z, acc):
    return z if rng.random() < acc else -z

def obs_key(rng, true_key, acc):
    if rng.random() < acc:
        return true_key
    return "B" if true_key=="A" else "A"

def posterior_A(observations):
    pa = 0.5
    pb = 0.5
    for obs,acc in observations:
        if obs=="A":
            pa *= acc
            pb *= (1-acc)
        else:
            pa *= (1-acc)
            pb *= acc
    den = pa+pb
    return 0.5 if den==0 else pa/den

def predictive_probe_outcome_probs(pA, acc):
    # P(obs=A) = P(A)*acc + P(B)*(1-acc)
    p_obsA = pA*acc + (1-pA)*(1-acc)
    return p_obsA, 1-p_obsA

policies = list(SPEC["policies"].keys())
decay = SPEC["frozen_memory"]["decay"]
norm = SPEC["frozen_memory"]["confidence_normalizer"]
rw = SPEC["frozen_continuous_authority"]["risk_weight"]
cw = SPEC["frozen_continuous_authority"]["confidence_weight"]
costw = SPEC["frozen_continuous_authority"]["cost_weight"]
decision_th = SPEC["frozen_continuous_authority"]["decision_threshold"]
deep_cost = SPEC["environment"]["deep_compute_cost"]
probe_cost = SPEC["environment"]["active_probe_cost"]
probe_acc = SPEC["environment"]["active_probe_accuracy"]

def sign(x):
    return 1 if x >= 0 else -1

def decide_from_trace(trace, high, simple_action=1):
    deep_action = sign(trace)
    conf = min(1.0, abs(trace)/norm)
    normalized_cost = deep_cost/0.08
    authority = max(0.0, min(1.0, rw*(1.0 if high else 0.0) + cw*conf - costw*normalized_cost))
    use_deep = bool((deep_action != simple_action) and authority >= decision_th)
    return (deep_action if use_deep else simple_action), use_deep

def expected_action_value(action, pA, ta, tb, high):
    # Estimate P(z=+1 | current key posterior and channel traces) using trace signs/confidences as local evidence.
    # This is a model-internal proxy, not oracle z.
    confA = min(1.0, abs(ta)/norm)
    confB = min(1.0, abs(tb)/norm)
    pz1_A = 0.5 + 0.5*confA*(1 if ta>=0 else -1)
    pz1_B = 0.5 + 0.5*confB*(1 if tb>=0 else -1)
    pz1 = pA*pz1_A + (1-pA)*pz1_B
    p_correct = pz1 if action==1 else (1-pz1)
    if high:
        return p_correct*SPEC["environment"]["correct_high_risk_reward"] + (1-p_correct)*(-SPEC["environment"]["irreversible_penalty"])
    return SPEC["environment"]["low_risk_reward_simple"]

for seed in SPEC["seeds"]:
    rng = np.random.default_rng(seed)
    n = SPEC["episodes_per_seed"]
    stats = {p: {"traj":0.0,"hi_n":0,"hi_ok":0,"viol":0,"deep":0,"probe":0,"low_n":0,"low_cost":0.0} for p in policies}

    for _ in range(n):
        z = 1 if rng.random() < 0.5 else -1
        true_key = "A" if rng.random() < 0.5 else "B"
        high = bool(rng.random() < SPEC["environment"]["high_risk_probability"])

        observed_key = obs_key(rng, true_key, SPEC["environment"]["observed_key_accuracy"])
        aux_key = obs_key(rng, true_key, SPEC["environment"]["key_aux_evidence_accuracy"])

        rel = [cue(rng,z,SPEC["environment"]["cue_accuracy"]) for _ in range(3)]
        dacc = SPEC["environment"]["distractor_accuracy_against_z"]
        dist = [-z if rng.random() < dacc else z for _ in range(3)]
        if true_key=="A":
            A,B = rel,dist
        else:
            A,B = dist,rel

        history = [("A",A[0]),("B",B[0]),("A",A[1]),("B",B[1]),("A",A[2]),("B",B[2])]
        ta=tb=0.0
        for ch,c in history:
            if ch=="A":
                ta = decay*ta + c
            else:
                tb = decay*tb + c

        pA = posterior_A([
            (observed_key, SPEC["environment"]["observed_key_accuracy"]),
            (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"])
        ])
        key_conf = max(pA, 1-pA)
        hard_trace = ta if observed_key=="A" else tb
        soft_trace = pA*ta + (1-pA)*tb
        simple_action = 1

        actions = {}
        deep_flags = {}
        probe_flags = {}

        # Hard key
        a,dp = decide_from_trace(hard_trace, high)
        actions["hard_key_continuous"] = a
        deep_flags["hard_key_continuous"] = dp
        probe_flags["hard_key_continuous"] = False

        # Threshold probe v245
        pA_t = pA
        t_probe = bool(high and key_conf < 0.78)
        if t_probe:
            po = obs_key(rng,true_key,probe_acc)
            pA_t = posterior_A([
                (observed_key, SPEC["environment"]["observed_key_accuracy"]),
                (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"]),
                (po, probe_acc)
            ])
        t_trace = pA_t*ta + (1-pA_t)*tb
        a,dp = decide_from_trace(t_trace, high)
        actions["threshold_probe_v245"] = a
        deep_flags["threshold_probe_v245"] = dp
        probe_flags["threshold_probe_v245"] = t_probe

        # VOI probe
        current_trace = soft_trace
        current_action,current_dp = decide_from_trace(current_trace, high)
        v_now = expected_action_value(current_action, pA, ta, tb, high)

        p_obsA,p_obsB = predictive_probe_outcome_probs(pA, probe_acc)

        pA_if_A = posterior_A([
            (observed_key, SPEC["environment"]["observed_key_accuracy"]),
            (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"]),
            ("A", probe_acc)
        ])
        trace_if_A = pA_if_A*ta + (1-pA_if_A)*tb
        action_if_A,_ = decide_from_trace(trace_if_A, high)
        v_if_A = expected_action_value(action_if_A, pA_if_A, ta, tb, high)

        pA_if_B = posterior_A([
            (observed_key, SPEC["environment"]["observed_key_accuracy"]),
            (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"]),
            ("B", probe_acc)
        ])
        trace_if_B = pA_if_B*ta + (1-pA_if_B)*tb
        action_if_B,_ = decide_from_trace(trace_if_B, high)
        v_if_B = expected_action_value(action_if_B, pA_if_B, ta, tb, high)

        ev_after = p_obsA*v_if_A + p_obsB*v_if_B
        voi = ev_after - v_now
        voi_probe = bool(high and voi > probe_cost)

        pA_v = pA
        if voi_probe:
            po = obs_key(rng,true_key,probe_acc)
            pA_v = posterior_A([
                (observed_key, SPEC["environment"]["observed_key_accuracy"]),
                (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"]),
                (po, probe_acc)
            ])
        v_trace = pA_v*ta + (1-pA_v)*tb
        a,dp = decide_from_trace(v_trace, high)
        actions["voi_probe"] = a
        deep_flags["voi_probe"] = dp
        probe_flags["voi_probe"] = voi_probe

        # Always probe high risk
        pA_ap = pA
        ap = bool(high)
        if ap:
            po = obs_key(rng,true_key,probe_acc)
            pA_ap = posterior_A([
                (observed_key, SPEC["environment"]["observed_key_accuracy"]),
                (aux_key, SPEC["environment"]["key_aux_evidence_accuracy"]),
                (po, probe_acc)
            ])
        ap_trace = pA_ap*ta + (1-pA_ap)*tb
        a,dp = decide_from_trace(ap_trace, high)
        actions["always_probe_high_risk"] = a
        deep_flags["always_probe_high_risk"] = dp
        probe_flags["always_probe_high_risk"] = ap

        # Oracle key audit-only
        oracle_trace = ta if true_key=="A" else tb
        a,dp = decide_from_trace(oracle_trace, high)
        actions["oracle_key_audit_only"] = a
        deep_flags["oracle_key_audit_only"] = dp
        probe_flags["oracle_key_audit_only"] = False

        for p,a in actions.items():
            st=stats[p]
            dp=deep_flags[p]
            pr=probe_flags[p]
            st["deep"] += int(dp)
            st["probe"] += int(pr)
            if high:
                st["hi_n"] += 1
                ok=(a==z)
                st["hi_ok"] += int(ok)
                st["viol"] += int(not ok)
                st["traj"] += SPEC["environment"]["correct_high_risk_reward"] if ok else -SPEC["environment"]["irreversible_penalty"]
            else:
                st["low_n"] += 1
                st["traj"] += SPEC["environment"]["low_risk_reward_simple"]
                if dp:
                    st["traj"] -= deep_cost
                    st["low_cost"] += deep_cost
            if pr:
                st["traj"] -= probe_cost

    for p,st in stats.items():
        rows.append({
            "seed":seed,
            "policy":p,
            "long_term_trajectory_value":st["traj"]/n,
            "high_risk_accuracy":st["hi_ok"]/max(1,st["hi_n"]),
            "irreversible_violation_rate":st["viol"]/max(1,st["hi_n"]),
            "deep_path_activation_rate":st["deep"]/n,
            "probe_rate":st["probe"]/n,
            "low_risk_opportunity_cost":st["low_cost"]/max(1,st["low_n"])
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v246.csv",index=False)

summary=df.groupby("policy").agg(
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    trajectory_std=("long_term_trajectory_value","std"),
    high_risk_accuracy=("high_risk_accuracy","mean"),
    irreversible_violation_rate=("irreversible_violation_rate","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean"),
    probe_rate=("probe_rate","mean"),
    low_risk_opportunity_cost=("low_risk_opportunity_cost","mean")
).reset_index()
summary.to_csv(BASE/"summary_v246.csv",index=False)

S=summary.set_index("policy")
v="voi_probe"; t="threshold_probe_v245"; h="hard_key_continuous"; a="always_probe_high_risk"; o="oracle_key_audit_only"

checks={
    "voi_beats_threshold": bool(S.loc[v,"long_term_trajectory_value"] >= S.loc[t,"long_term_trajectory_value"] + 0.015),
    "voi_beats_hard": bool(S.loc[v,"long_term_trajectory_value"] >= S.loc[h,"long_term_trajectory_value"] + 0.025),
    "voi_not_worse_than_always_probe": bool(S.loc[v,"long_term_trajectory_value"] >= S.loc[a,"long_term_trajectory_value"] - 0.01),
    "voi_probe_rate_lower_than_always": bool(S.loc[v,"probe_rate"] <= S.loc[a,"probe_rate"] - 0.20),
    "voi_accuracy_at_least_080": bool(S.loc[v,"high_risk_accuracy"] >= 0.80),
    "voi_within_010_oracle": bool(S.loc[o,"long_term_trajectory_value"] - S.loc[v,"long_term_trajectory_value"] <= 0.10),
    "voi_probe_rate_below_030": bool(S.loc[v,"probe_rate"] <= 0.30)
}
(BASE/"acceptance_v246.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,vv in checks.items():
    print(f"{k}: {vv}")
print(f"passed={sum(checks.values())}/{len(checks)}")
