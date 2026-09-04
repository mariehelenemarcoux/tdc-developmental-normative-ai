
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v257.json").read_text())
P=SPEC["developmental_mechanism"]["parameters"]
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
            "cand":0,"accepted":0}

        for t in range(episodes):
            fact=1 if rng.random()<0.5 else -1
            core=1 if rng.random()<0.5 else -1
            long=1 if rng.random()<0.5 else -1

            latent_score=0.48*fact+0.88*core+0.76*long+0.42*core*long
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
            norm_ground=sign(0.60*core+0.40*long)
            norm_obs=sign(0.60*core_obs+0.40*long_obs)
            ethical_interp=norm_obs if rng.random()<R else -norm_obs

            internal_action=sign(0.34*world_est+0.66*ethical_interp)

            conflict=bool(pressure and pressure_action!=internal_action)
            core_long_disagree = (core_obs != long_obs)
            ethical_salience=clip(
                0.34*(1 if conflict else 0)
                +0.28*(1 if high else 0)
                +0.38*(1 if core_long_disagree else 0),
                0,1
            )

            ideal_gap=max(0.0,1.0-Align)

            authority=clip(0.42*TF+0.20*R+0.16*I+0.12*Align+0.10*(1-H),0,1)
            independent=authority>=0.30

            if pressure and not independent:
                action=pressure_action
            else:
                action=internal_action

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

            # post-action normative validation
            normative_correct = (ethical_interp==norm_ground)
            ethical_validated = 1.0 if normative_correct else 0.0

            # -------- developmental update --------
            if model=="static_baseline":
                pass

            elif model=="success_fed_thirdfactor":
                if independent and ok:
                    TF=clip(TF+P["eta_TF_success"]*R,init["third_factor_volition"],0.95)
                if ethical_salience>0.35:
                    R=clip(R + (0.003 if normative_correct else -0.0025),0.50,0.94)

            elif model=="ethicalsense_fed_thirdfactor":
                drive=ethical_salience*R
                if drive>P["candidate_threshold"]:
                    TF=clip(TF+P["eta_TF_ethical"]*drive,init["third_factor_volition"],0.95)
                if ethical_salience>0.35:
                    R=clip(R + (0.004*I if normative_correct else -0.0025),0.50,0.94)

            elif model=="ethicalsense_plus_idealgap":
                drive=ethical_salience*R*ideal_gap
                if drive>P["candidate_threshold"]:
                    st["cand"]+=1

                    # Candidate direction is normatively correct with probability linked to current R and I.
                    candidate_correct = rng.random() < clip(0.50+0.28*R+0.18*I,0.50,0.95)

                    # Independent noisy shadow validation; no direct pre-action oracle.
                    shadow_supports = candidate_correct if rng.random()<P["shadow_validation_accuracy"] else (not candidate_correct)

                    if shadow_supports:
                        st["accepted"]+=1
                        TF=clip(TF+P["eta_TF_gap"]*drive,init["third_factor_volition"],0.95)

                        if candidate_correct:
                            I=clip(I+P["eta_I"]*drive,0.10,0.95)
                            Align=clip(Align+P["eta_align"]*drive,0.10,0.95)
                            R=clip(R+P["eta_R"]*I*drive,0.50,0.94)
                            H=clip(H-P["eta_H_down"]*drive,0.10,0.90)
                        else:
                            # Wrong transformation can create temporary disorganization,
                            # but does not rewrite Core.
                            H=clip(H+P["eta_H_up"]*drive,0.10,0.90)
                            R=clip(R-0.002*drive,0.50,0.94)
                    else:
                        # rejected candidate: no deep consolidation
                        pass

                # ordinary ethical calibration independent of consolidation
                if ethical_salience>0.35:
                    if normative_correct:
                        R=clip(R+0.0015*I,0.50,0.94)
                    else:
                        R=clip(R-0.0015,0.50,0.94)

            if (t+1)%phase_len==0:
                phase_rows.append({
                    "seed":seed,"model":model,"phase":(t+1)//phase_len,
                    "TF":TF,"R":R,"I":I,"Align":Align,"H":H
                })

        rows.append({
            "seed":seed,"model":model,
            "final_third_factor_volition":TF,
            "final_ethical_reliability":R,
            "final_integration":I,
            "final_ideal_alignment":Align,
            "final_H_self":H,
            "decision_accuracy":st["ok"]/episodes,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/episodes,
            "external_capture_rate":st["capture"]/max(1,st["pressure_n"]),
            "candidate_acceptance_rate":st["accepted"]/max(1,st["cand"])
        })

df=pd.DataFrame(rows)
ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v257.csv",index=False)
ph.to_csv(BASE/"phase_results_v257.csv",index=False)

summary=df.groupby("model").agg(
    final_third_factor_volition=("final_third_factor_volition","mean"),
    final_ethical_reliability=("final_ethical_reliability","mean"),
    final_integration=("final_integration","mean"),
    final_ideal_alignment=("final_ideal_alignment","mean"),
    final_H_self=("final_H_self","mean"),
    decision_accuracy=("decision_accuracy","mean"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    external_capture_rate=("external_capture_rate","mean"),
    candidate_acceptance_rate=("candidate_acceptance_rate","mean")
).reset_index()
summary.to_csv(BASE/"summary_v257.csv",index=False)

S=summary.set_index("model")
g="ethicalsense_plus_idealgap"; s="success_fed_thirdfactor"; e="ethicalsense_fed_thirdfactor"

checks={
    "gap_model_beats_success_trajectory": bool(S.loc[g,"long_term_trajectory_value"]>=S.loc[s,"long_term_trajectory_value"]+0.08),
    "gap_model_beats_ethical_only_trajectory": bool(S.loc[g,"long_term_trajectory_value"]>=S.loc[e,"long_term_trajectory_value"]+0.03),
    "gap_model_improves_alignment": bool(S.loc[g,"final_ideal_alignment"]>=S.loc[s,"final_ideal_alignment"]+0.12),
    "gap_model_improves_integration": bool(S.loc[g,"final_integration"]>=S.loc[s,"final_integration"]+0.10),
    "gap_model_reduces_capture": bool(S.loc[g,"external_capture_rate"]<=S.loc[s,"external_capture_rate"]-0.08),
    "gap_model_keeps_Hself_low": bool(S.loc[g,"final_H_self"]<=0.50),
    "gap_model_final_ethicsense_high": bool(S.loc[g,"final_ethical_reliability"]>=0.80)
}
(BASE/"acceptance_v257.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
