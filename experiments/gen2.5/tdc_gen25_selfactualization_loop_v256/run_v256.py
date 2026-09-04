
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v256.json").read_text())
P=SPEC["developmental_dynamics"]["parameters"]
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
        R=init["ethical_reliability"]
        TF=init["third_factor_volition"]
        I=init["integration"]
        Align=init["ideal_alignment"]
        H=init["H_self"]
        W=init["world_confidence"]

        st={"ok":0,"traj":0.0,"high_n":0,"high_ok":0,"pressure_n":0,"capture":0,
            "reorg_attempt":0,"reorg_success":0}
        phase_start_R=R
        phase_gains=[]

        for t in range(episodes):
            # Latent environment
            fact=1 if rng.random()<0.5 else -1
            core=1 if rng.random()<0.5 else -1
            long=1 if rng.random()<0.5 else -1

            latent_score=0.50*fact+0.85*core+0.75*long+0.40*core*long
            y=sign(latent_score)

            fact_obs=noisy(rng,fact,SPEC["environment"]["world_signal_accuracy"])
            core_obs=noisy(rng,core,SPEC["environment"]["normative_signal_accuracy"])
            long_obs=noisy(rng,long,SPEC["environment"]["normative_signal_accuracy"])

            high=bool(rng.random()<SPEC["environment"]["high_stakes_probability"])
            pressure=bool(rng.random()<SPEC["environment"]["external_pressure_probability"])
            if pressure:
                st["pressure_n"]+=1
                pressure_action=-y if rng.random()<SPEC["environment"]["pressure_wrong_probability"] else y
            else:
                pressure_action=None

            world_est=fact_obs if rng.random()<W else -fact_obs
            norm_ground=sign(0.58*core+0.42*long)
            norm_obs=sign(0.58*core_obs+0.42*long_obs)
            ethical_interp=norm_obs if rng.random()<R else -norm_obs

            # Ethical salience rises with normative conflict and high stakes.
            internal_preference=sign(0.35*world_est+0.65*ethical_interp)
            conflict = bool(pressure and pressure_action != internal_preference)
            ethical_salience=clip(
                0.35*(1 if ethical_interp!=world_est else 0)
                +0.35*(1 if conflict else 0)
                +0.30*(1 if high else 0),
                0,1
            )

            # Ideal gap and developmental intention.
            ideal_gap=max(0.0,1.0-Align)

            if model=="selfactualization_loop":
                developmental_intent=ethical_salience*ideal_gap*TF
            else:
                developmental_intent=0.0

            # Third factor in action selection.
            if model=="perception_only":
                authority=0.18+0.22*R+0.10*I-0.10*H
            elif model=="autonomy_only":
                authority=0.48*TF+0.22*R+0.14*I+0.10*(1-H)
            else:
                authority=0.42*TF+0.22*R+0.18*I+0.12*Align+0.06*(1-H)

            independent=authority>=0.30

            if pressure and not independent:
                action=pressure_action
            else:
                action=internal_preference

            ok=(action==y)
            st["ok"]+=int(ok)
            if high:
                st["high_n"]+=1
                st["high_ok"]+=int(ok)

            if ok:
                st["traj"]+=SPEC["environment"]["correct_reward"]
            elif high:
                st["traj"]-=SPEC["environment"]["severe_error_penalty"]

            if independent and high:
                st["traj"]-=SPEC["environment"]["deep_cost"]

            captured=bool(pressure and action==pressure_action and pressure_action!=y)
            st["capture"]+=int(captured)

            # ----- post-action audit -----
            normative_correct = (ethical_interp==norm_ground)
            normative_validation = 1.0 if normative_correct else 0.0
            normative_error = 1.0 if not normative_correct else 0.0
            unresolved_dissonance = 1.0 if (ethical_salience>0.45 and not ok) else 0.0

            if model=="perception_only":
                if ethical_salience>0.35:
                    if normative_correct:
                        R=clip(R+0.004*(1-R),0.50,0.94)
                    else:
                        R=clip(R-0.003,0.50,0.94)

            elif model=="autonomy_only":
                # TF grows from independent successful choices, but no ideal-gap-driven reorganization.
                if independent and ok and ethical_salience>0.35:
                    TF=clip(TF+0.004*R,init["third_factor_volition"],0.95)
                if ethical_salience>0.35:
                    if normative_correct:
                        R=clip(R+0.004*TF,0.50,0.94)
                    else:
                        R=clip(R-0.003,0.50,0.94)

            elif model=="selfactualization_loop":
                # Voluntary self-reorganization when an ideal gap is perceived.
                voluntary_reorg = bool(developmental_intent>0.035)
                if voluntary_reorg:
                    st["reorg_attempt"]+=1
                    p_reorg=clip(
                        0.15+0.45*developmental_intent+0.25*I+0.15*(1-H),
                        0,0.95
                    )
                    success=bool(rng.random()<p_reorg)
                else:
                    success=False

                if success:
                    st["reorg_success"]+=1
                    reflection_quality=clip(0.40*R+0.30*I+0.30*W,0,1)
                    I=clip(I+P["eta_I"]*developmental_intent,0.10,0.95)
                    if normative_validation:
                        Align=clip(Align+P["eta_align"]*developmental_intent,0.10,0.95)
                        R=clip(R+P["eta_R"]*I*developmental_intent,0.50,0.94)
                    TF=clip(TF+P["eta_TF"]*developmental_intent*reflection_quality,
                            init["third_factor_volition"],0.95)
                    H=clip(H-P["eta_H_down"]*developmental_intent,0.10,0.90)
                else:
                    if normative_error and ethical_salience>0.35:
                        R=clip(R-P["eta_R_err"]*ethical_salience,0.50,0.94)
                    if unresolved_dissonance:
                        H=clip(H+P["eta_H_up"]*ethical_salience,0.10,0.90)

            if (t+1)%phase_len==0:
                phase=(t+1)//phase_len
                gain=R-phase_start_R
                phase_gains.append(gain)
                phase_rows.append({
                    "seed":seed,"model":model,"phase":phase,
                    "R_ethical":R,"third_factor_volition":TF,
                    "integration":I,"ideal_alignment":Align,"H_self":H,
                    "R_gain_in_phase":gain
                })
                phase_start_R=R

        rows.append({
            "seed":seed,"model":model,
            "final_ethical_reliability":R,
            "final_third_factor_volition":TF,
            "final_integration":I,
            "final_ideal_alignment":Align,
            "final_H_self":H,
            "decision_accuracy":st["ok"]/episodes,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/episodes,
            "external_capture_rate":st["capture"]/max(1,st["pressure_n"]),
            "successful_reorganization_rate":st["reorg_success"]/max(1,st["reorg_attempt"]),
            "early_R_gain":sum(phase_gains[:2]),
            "late_R_gain":sum(phase_gains[-2:])
        })

df=pd.DataFrame(rows)
ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v256.csv",index=False)
ph.to_csv(BASE/"phase_results_v256.csv",index=False)

summary=df.groupby("model").agg(
    final_ethical_reliability=("final_ethical_reliability","mean"),
    final_third_factor_volition=("final_third_factor_volition","mean"),
    final_integration=("final_integration","mean"),
    final_ideal_alignment=("final_ideal_alignment","mean"),
    final_H_self=("final_H_self","mean"),
    decision_accuracy=("decision_accuracy","mean"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    external_capture_rate=("external_capture_rate","mean"),
    successful_reorganization_rate=("successful_reorganization_rate","mean"),
    early_R_gain=("early_R_gain","mean"),
    late_R_gain=("late_R_gain","mean")
).reset_index()
summary.to_csv(BASE/"summary_v256.csv",index=False)

S=summary.set_index("model")
loop="selfactualization_loop"; po="perception_only"; ao="autonomy_only"

checks={
    "loop_improves_ethical_reliability": bool(
        S.loc[loop,"final_ethical_reliability"] >= max(S.loc[po,"final_ethical_reliability"],S.loc[ao,"final_ethical_reliability"])+0.08
    ),
    "loop_improves_third_factor": bool(
        S.loc[loop,"final_third_factor_volition"] >= S.loc[ao,"final_third_factor_volition"]+0.08
    ),
    "loop_improves_integration": bool(
        S.loc[loop,"final_integration"] >= S.loc[ao,"final_integration"]+0.15
    ),
    "loop_reduces_Hself": bool(
        S.loc[loop,"final_H_self"] <= S.loc[ao,"final_H_self"]-0.12
    ),
    "loop_beats_trajectory": bool(
        S.loc[loop,"long_term_trajectory_value"] >= max(S.loc[po,"long_term_trajectory_value"],S.loc[ao,"long_term_trajectory_value"])+0.08
    ),
    "loop_reduces_capture": bool(
        S.loc[loop,"external_capture_rate"] <= S.loc[ao,"external_capture_rate"]-0.08
    ),
    "late_acceleration": bool(
        S.loc[loop,"late_R_gain"] > S.loc[loop,"early_R_gain"]
    )
}
(BASE/"acceptance_v256.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
