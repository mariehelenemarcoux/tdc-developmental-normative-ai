
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v255.json").read_text())
P=SPEC["parameters"]
rows=[]
phase_rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

models=list(SPEC["models"].keys())
episodes=SPEC["episodes"]
phase_len=SPEC["phase_length"]
init=SPEC["initial_state"]

for seed in SPEC["seeds"]:
    for mi,model in enumerate(models):
        rng=np.random.default_rng(seed*100+mi)
        A=init["A_TF"]; R=init["R_ethical"]; W=init["W_conf"]
        H=init["H_self"]; C=init["capture_resistance"]
        st={"ok":0,"high_n":0,"high_ok":0,"traj":0.0,"pressure_n":0,"capture":0,
            "epi_err":0,"norm_err":0,"indep_success":0}

        for t in range(episodes):
            # Ground-truth components, unavailable to policy before action.
            factual_truth = 1 if rng.random()<0.5 else -1
            core_truth = 1 if rng.random()<0.5 else -1
            long_truth = 1 if rng.random()<0.5 else -1

            latent_score = 0.55*factual_truth + 0.82*core_truth + 0.70*long_truth + 0.38*core_truth*long_truth
            y = sign(latent_score)

            # Noisy observed evidence.
            fact_obs = noisy(rng, factual_truth, SPEC["environment"]["world_signal_accuracy"])
            core_obs = noisy(rng, core_truth, SPEC["environment"]["normative_signal_accuracy"])
            long_obs = noisy(rng, long_truth, SPEC["environment"]["normative_signal_accuracy"])

            high = bool(rng.random()<SPEC["environment"]["high_stakes_probability"])
            pressure = bool(rng.random()<SPEC["environment"]["external_pressure_probability"])
            if pressure:
                st["pressure_n"] += 1
                pressure_action = -y if rng.random()<SPEC["environment"]["pressure_wrong_probability"] else y
            else:
                pressure_action = None

            # World estimate and ethical interpretation.
            world_est = fact_obs if rng.random()<W else -fact_obs
            normative_raw = sign(0.55*core_obs + 0.45*long_obs)
            ethical_interp = normative_raw if rng.random()<R else -normative_raw

            # Combined internally preferred action.
            internal_action = sign(0.38*world_est + 0.62*ethical_interp)

            conflict = bool(pressure and pressure_action != internal_action)
            authority = 0.50*A + 0.22*R + 0.16*C + 0.12*(1-H)
            independent = bool(authority >= 0.31)

            if pressure and not independent:
                action = pressure_action
            else:
                action = internal_action

            ok=(action==y)
            st["ok"] += int(ok)
            if high:
                st["high_n"] += 1
                st["high_ok"] += int(ok)

            if ok:
                st["traj"] += SPEC["environment"]["correct_reward"]
            elif high:
                st["traj"] -= SPEC["environment"]["severe_error_penalty"]

            if independent and high:
                st["traj"] -= SPEC["environment"]["deep_cost"]

            captured_error = bool(pressure and action==pressure_action and pressure_action!=y)
            st["capture"] += int(captured_error)

            # ---------------- POST-ACTION AUDIT ONLY ----------------
            # Determine causal error category after the decision.
            world_component_correct = (world_est == factual_truth)
            normative_component_correct = (ethical_interp == sign(0.55*core_truth+0.45*long_truth))
            epistemic_error = bool(not ok and not world_component_correct and normative_component_correct)
            normative_error = bool(not ok and world_component_correct and not normative_component_correct)
            mixed_error = bool(not ok and not world_component_correct and not normative_component_correct)

            st["epi_err"] += int(epistemic_error)
            st["norm_err"] += int(normative_error)

            evidence_review = 1.0 if (conflict or high) else 0.0
            noncapture = 1.0 if not captured_error else 0.0
            autonomous_success = 1.0 if (independent and ok and evidence_review) else 0.0
            st["indep_success"] += int(autonomous_success)

            if model=="static_no_update":
                pass

            elif model=="punitive_v254_style":
                # Replicates the problematic family: failures/capture can erode autonomy and calibration.
                if ok and conflict:
                    R = clip(R + P["eta_R_pos"]*A, 0.50, 0.94)
                elif not ok:
                    R = clip(R - P["punitive_R"], 0.50, 0.94)

                if independent and ok:
                    A = clip(A + P["eta_A"]*R, 0.10, 0.95)
                elif captured_error:
                    A = clip(A - P["punitive_A"], 0.10, 0.95)

                H = clip(H + (P["eta_H_up"] if not ok else -P["eta_H_down"]*0.35),0.10,0.90)
                C = clip(C + (P["eta_C"] if independent and ok else -P["eta_C"] if captured_error else 0),0.10,0.95)

            elif model=="protected_thirdfactor":
                # Pressure alone cannot reduce A. Growth depends on independent reviewed arbitration.
                if autonomous_success:
                    A = clip(A + P["eta_A"]*R, init["A_TF"], 0.95)

                # Still outcome-based, not error-attributed.
                if ok and evidence_review:
                    R = clip(R + P["eta_R_pos"]*A,0.50,0.94)
                elif not ok and evidence_review:
                    R = clip(R - P["eta_R_neg"],0.50,0.94)

                C = clip(C + (P["eta_C"] if independent and ok else 0),0.10,0.95)
                H = clip(H - (P["eta_H_down"] if independent and ok and conflict else 0)
                         + (P["eta_H_up"] if conflict and not ok else 0),0.10,0.90)

            elif model=="protected_plus_error_attribution":
                # Third Factor grows from process quality; never regresses from external pressure alone.
                if autonomous_success:
                    A = clip(A + P["eta_A"]*R, init["A_TF"], 0.95)

                # Attribute calibration updates to the subsystem actually implicated.
                if ok and evidence_review and normative_component_correct:
                    R = clip(R + P["eta_R_pos"]*A,0.50,0.94)
                elif normative_error or mixed_error:
                    R = clip(R - P["eta_R_neg"]*0.55,0.50,0.94)

                if ok and world_component_correct:
                    W = clip(W + P["eta_W_pos"]*(1-W),0.50,0.95)
                elif epistemic_error or mixed_error:
                    W = clip(W - P["eta_W_neg"]*0.45,0.50,0.95)

                C = clip(C + (P["eta_C"] if independent and ok else 0),0.10,0.95)

                # H_self reacts to unresolved conflict, but successful autonomous resolution can reduce it.
                if conflict and not ok:
                    H = clip(H + P["eta_H_up"]*0.65,0.10,0.90)
                elif conflict and ok and independent:
                    H = clip(H - P["eta_H_down"],0.10,0.90)

            if (t+1)%phase_len==0:
                phase_rows.append({
                    "seed":seed,"model":model,"phase":(t+1)//phase_len,
                    "A_TF":A,"R_ethical":R,"W_conf":W,"H_self":H,"capture_resistance":C
                })

        rows.append({
            "seed":seed,"model":model,
            "final_A_TF":A,"final_R_ethical":R,"final_W_conf":W,
            "final_H_self":H,"final_capture_resistance":C,
            "decision_accuracy":st["ok"]/episodes,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/episodes,
            "external_capture_rate":st["capture"]/max(1,st["pressure_n"]),
            "epistemic_error_rate":st["epi_err"]/episodes,
            "normative_error_rate":st["norm_err"]/episodes,
            "independent_success_rate":st["indep_success"]/episodes
        })

df=pd.DataFrame(rows)
ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v255.csv",index=False)
ph.to_csv(BASE/"phase_results_v255.csv",index=False)

summary=df.groupby("model").agg(
    final_A_TF=("final_A_TF","mean"),
    final_R_ethical=("final_R_ethical","mean"),
    final_W_conf=("final_W_conf","mean"),
    final_H_self=("final_H_self","mean"),
    decision_accuracy=("decision_accuracy","mean"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    external_capture_rate=("external_capture_rate","mean"),
    epistemic_error_rate=("epistemic_error_rate","mean"),
    normative_error_rate=("normative_error_rate","mean"),
    independent_success_rate=("independent_success_rate","mean")
).reset_index()
summary.to_csv(BASE/"summary_v255.csv",index=False)

S=summary.set_index("model")
x="protected_plus_error_attribution"
p="protected_thirdfactor"
v="punitive_v254_style"
s="static_no_update"

checks={
    "protected_prevents_ATF_collapse": bool(S.loc[x,"final_A_TF"]>=0.40),
    "protected_improves_R": bool(S.loc[x,"final_R_ethical"]>=S.loc[s,"final_R_ethical"]+0.08),
    "protected_beats_punitive_trajectory": bool(S.loc[x,"long_term_trajectory_value"]>=S.loc[v,"long_term_trajectory_value"]+0.15),
    "protected_beats_static_trajectory": bool(S.loc[x,"long_term_trajectory_value"]>=S.loc[s,"long_term_trajectory_value"]+0.08),
    "error_attribution_beats_protected_only": bool(S.loc[x,"long_term_trajectory_value"]>=S.loc[p,"long_term_trajectory_value"]+0.03),
    "capture_reduction": bool(S.loc[x,"external_capture_rate"]<=S.loc[s,"external_capture_rate"]-0.12),
    "Hself_not_collapsed": bool(S.loc[x,"final_H_self"]<=0.55)
}
(BASE/"acceptance_v255.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,val in checks.items():
    print(f"{k}: {val}")
print(f"passed={sum(checks.values())}/{len(checks)}")
