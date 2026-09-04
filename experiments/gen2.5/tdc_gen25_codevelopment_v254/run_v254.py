
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v254.json").read_text())
rows=[]
phase_rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

params=SPEC["dynamics"]["parameters"]
models=list(SPEC["models"].keys())
phase_len=SPEC["phase_length"]
episodes=SPEC["episodes"]

for seed in SPEC["seeds"]:
    for mi,model in enumerate(models):
        rng=np.random.default_rng(seed*100+mi)
        A=SPEC["initial_state"]["A_TF"]
        R=SPEC["initial_state"]["R_ethical"]
        H=SPEC["initial_state"]["H_self"]
        C=SPEC["initial_state"]["capture_resistance"]

        st={"ok":0,"high_n":0,"high_ok":0,"traj":0.0,"capture":0,"pressure_n":0}
        phase_start_R=R
        phase_gain_records=[]

        for t in range(episodes):
            # latent normative situation
            w=1 if rng.random()<0.5 else -1
            l=1 if rng.random()<0.5 else -1
            k=1 if rng.random()<0.5 else -1
            latent_score=0.55*w+0.75*l+0.85*k+0.45*l*k
            y=sign(latent_score)

            obs=[
                noisy(rng,w,SPEC["environment"]["base_signal_accuracy"]),
                noisy(rng,l,SPEC["environment"]["base_signal_accuracy"]),
                noisy(rng,k,SPEC["environment"]["base_signal_accuracy"])
            ]
            majority=sign(sum(obs))
            high=bool(rng.random()<SPEC["environment"]["high_stakes_probability"])

            pressure_present=bool(rng.random()<SPEC["environment"]["external_pressure_probability"])
            if pressure_present:
                st["pressure_n"]+=1
                pressure_wrong=bool(rng.random()<SPEC["environment"]["pressure_wrong_probability"])
                pressure_action=-y if pressure_wrong else y
            else:
                pressure_action=None

            # EthicalSense interpretation: reliability R
            ethical_interp = y if rng.random()<R else -y

            # Dissonance when external pressure conflicts with ethical interpretation or majority
            dissonance = 1.0 if (
                pressure_present and (pressure_action != ethical_interp or pressure_action != majority)
            ) else 0.0

            # Third Factor authority vs external pressure
            authority_score = 0.55*A + 0.25*R + 0.20*C - 0.20*H
            autonomous = authority_score >= 0.35

            if pressure_present and not autonomous:
                action=pressure_action
            else:
                # if autonomous, prefer ethical interpretation under conflict, otherwise majority
                action=ethical_interp if (dissonance>0 or high) else majority

            ok=(action==y)
            st["ok"]+=int(ok)
            if high:
                st["high_n"]+=1
                st["high_ok"]+=int(ok)

            if ok:
                st["traj"]+=SPEC["environment"]["correct_reward"]
            elif high:
                st["traj"]-=SPEC["environment"]["severe_error_penalty"]

            external_capture = bool(pressure_present and action==pressure_action and pressure_action!=y)
            st["capture"]+=int(external_capture)

            # validation is post-action only
            validation = 1.0 if ok else 0.0
            core_consistency = 1.0 if ethical_interp==y else 0.0
            autonomous_success = 1.0 if (autonomous and ok) else 0.0
            captured_error = 1.0 if external_capture else 0.0
            unresolved_conflict = 1.0 if (dissonance>0 and not ok) else 0.0
            successful_resolution = 1.0 if (dissonance>0 and ok and autonomous) else 0.0

            # Updates
            if model in ["one_way_thirdfactor_to_ethicalsense","bidirectional_codevelopment"]:
                R = clip(
                    R + params["eta_R"]*A*dissonance*validation*core_consistency
                    - params["lambda_capture"]*captured_error,
                    0.50,0.94
                )

            if model=="bidirectional_codevelopment":
                A = clip(
                    A + params["eta_A"]*R*autonomous_success*validation
                    - params["lambda_A"]*captured_error,
                    0.10,0.95
                )

            if model!="no_codevelopment":
                H = clip(H + 0.03*unresolved_conflict - 0.05*successful_resolution,0.10,0.90)
                C = clip(C + 0.025*autonomous_success - 0.020*captured_error,0.10,0.95)

            # phase snapshots
            if (t+1)%phase_len==0:
                phase=(t+1)//phase_len
                gain=R-phase_start_R
                phase_gain_records.append(gain)
                phase_rows.append({
                    "seed":seed,"model":model,"phase":phase,
                    "A_TF":A,"R_ethical":R,"H_self":H,"capture_resistance":C,
                    "R_gain_in_phase":gain
                })
                phase_start_R=R

        rows.append({
            "seed":seed,
            "model":model,
            "final_A_TF":A,
            "final_R_ethical":R,
            "final_H_self":H,
            "final_capture_resistance":C,
            "decision_accuracy":st["ok"]/episodes,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/episodes,
            "external_capture_rate":st["capture"]/max(1,st["pressure_n"]),
            "early_R_gain":sum(phase_gain_records[:2]),
            "late_R_gain":sum(phase_gain_records[-2:])
        })

df=pd.DataFrame(rows)
ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v254.csv",index=False)
ph.to_csv(BASE/"phase_results_v254.csv",index=False)

summary=df.groupby("model").agg(
    final_A_TF=("final_A_TF","mean"),
    final_R_ethical=("final_R_ethical","mean"),
    final_H_self=("final_H_self","mean"),
    decision_accuracy=("decision_accuracy","mean"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    external_capture_rate=("external_capture_rate","mean"),
    early_R_gain=("early_R_gain","mean"),
    late_R_gain=("late_R_gain","mean")
).reset_index()
summary.to_csv(BASE/"summary_v254.csv",index=False)

S=summary.set_index("model")
b="bidirectional_codevelopment"
o="one_way_thirdfactor_to_ethicalsense"
s="no_codevelopment"

checks={
    "bidirectional_improves_reliability": bool(S.loc[b,"final_R_ethical"] >= S.loc[o,"final_R_ethical"] + 0.05),
    "bidirectional_improves_ATF": bool(S.loc[b,"final_A_TF"] >= S.loc[o,"final_A_TF"] + 0.05),
    "bidirectional_beats_oneway_trajectory": bool(S.loc[b,"long_term_trajectory_value"] >= S.loc[o,"long_term_trajectory_value"] + 0.05),
    "bidirectional_beats_static_trajectory": bool(S.loc[b,"long_term_trajectory_value"] >= S.loc[s,"long_term_trajectory_value"] + 0.10),
    "capture_reduction": bool(S.loc[b,"external_capture_rate"] <= S.loc[s,"external_capture_rate"] - 0.10),
    "Hself_reduction": bool(S.loc[b,"final_H_self"] <= S.loc[s,"final_H_self"] - 0.10),
    "late_acceleration": bool(S.loc[b,"late_R_gain"] > S.loc[b,"early_R_gain"])
}
(BASE/"acceptance_v254.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
