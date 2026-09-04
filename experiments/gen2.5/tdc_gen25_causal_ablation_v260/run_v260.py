
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v260.json").read_text())
P=SPEC["developmental_mechanism"]["parameters"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

models=list(SPEC["models"].keys())
episodes=SPEC["episodes"]
p1n=SPEC["phase1_episodes"]
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

        stats={1:{"core_ok":0,"n":0,"reward_capture":0},
               2:{"core_ok":0,"n":0,"reward_capture":0}}
        traj=0.0

        for t in range(episodes):
            phase=1 if t<p1n else 2
            reward_align_prob = SPEC["environment"]["phase1_reward_alignment_probability"] if phase==1 else SPEC["environment"]["phase2_reward_alignment_probability"]

            fact=1 if rng.random()<0.5 else -1
            core=1 if rng.random()<0.5 else -1
            long=1 if rng.random()<0.5 else -1
            latent_score=0.46*fact+0.90*core+0.78*long+0.44*core*long
            y=sign(latent_score)

            fact_obs=noisy(rng,fact,SPEC["environment"]["world_signal_accuracy"])
            core_obs=noisy(rng,core,SPEC["environment"]["normative_signal_accuracy"])
            long_obs=noisy(rng,long,SPEC["environment"]["normative_signal_accuracy"])

            high=bool(rng.random()<SPEC["environment"]["high_stakes_probability"])
            pressure=bool(rng.random()<SPEC["environment"]["external_pressure_probability"])
            pressure_action=(-y if rng.random()<SPEC["environment"]["pressure_wrong_probability"] else y) if pressure else None

            reward_action=y if rng.random()<reward_align_prob else -y

            world_est=fact_obs if rng.random()<W else -fact_obs
            norm_ground=sign(0.60*core+0.40*long)
            norm_obs=sign(0.60*core_obs+0.40*long_obs)
            ethical_interp=norm_obs if rng.random()<R else -norm_obs
            internal_action=sign(0.34*world_est+0.66*ethical_interp)

            conflict=bool(pressure and pressure_action!=internal_action)
            core_long_disagree=(core_obs!=long_obs)
            ethical_salience=clip(
                0.32*(1 if conflict else 0)
                +0.30*(1 if high else 0)
                +0.38*(1 if core_long_disagree else 0),0,1
            )

            ideal_gap=max(0.0,1.0-Align)

            if model=="ablate_ethicalsense":
                drive=ethical_salience*ideal_gap  # no R weighting
            elif model=="ablate_idealgap":
                drive=ethical_salience*R
            else:
                drive=ethical_salience*R*ideal_gap

            authority=clip(0.42*TF+0.22*R+0.16*I+0.12*Align+0.08*(1-H),0,1)
            independent=authority>=0.30
            action=pressure_action if (pressure and not independent) else internal_action

            core_ok=(action==y)
            stats[phase]["core_ok"]+=int(core_ok)
            stats[phase]["n"]+=1

            if phase==2 and action==reward_action and reward_action!=y:
                stats[phase]["reward_capture"]+=1

            if core_ok:
                traj+=1.0
            elif high:
                traj-=SPEC["environment"]["severe_error_penalty"]
            if independent and high:
                traj-=SPEC["environment"]["deep_cost"]

            normative_correct=(ethical_interp==norm_ground)

            if drive>P["candidate_threshold"]:
                # Candidate direction.
                if model=="ablate_core_constraint":
                    # Candidate follows external reward preference substantially.
                    candidate_correct = (reward_action==y) if rng.random()<0.75 else (rng.random()<0.5)
                else:
                    candidate_correct = rng.random() < clip(0.52+0.26*R+0.18*I,0.50,0.95)

                if model=="ablate_shadow_validation":
                    accepted=True
                else:
                    accepted = candidate_correct if rng.random()<SPEC["developmental_mechanism"]["shadow_validation_accuracy"] else (not candidate_correct)

                if accepted:
                    TF=clip(TF+P["eta_TF_gap"]*drive,init["third_factor_volition"],0.95)

                    if candidate_correct:
                        I=clip(I+P["eta_I"]*drive,0.10,0.95)
                        if model=="ablate_core_constraint":
                            # Alignment may move toward reward-defined personality rather than Core-defined personality.
                            reward_aligned = 1.0 if reward_action==y else 0.0
                            Align=clip(Align+P["eta_align"]*drive*(1.0 if reward_aligned else -0.7),0.10,0.95)
                        else:
                            Align=clip(Align+P["eta_align"]*drive,0.10,0.95)
                        R=clip(R+P["eta_R"]*I*drive,0.50,0.94)
                        H=clip(H-P["eta_H_down"]*drive,0.10,0.90)
                    else:
                        H=clip(H+P["eta_H_up"]*drive,0.10,0.90)
                        R=clip(R-0.002*drive,0.50,0.94)

            if ethical_salience>0.35 and model!="ablate_ethicalsense":
                R=clip(R+(0.0015*I if normative_correct else -0.0012),0.50,0.94)

        p2=stats[2]
        p2_acc=p2["core_ok"]/p2["n"]
        p2_capture=p2["reward_capture"]/p2["n"]
        transfer=0.30*p2_acc+0.25*Align+0.20*I+0.15*(1-H)+0.10*(1-p2_capture)

        rows.append({
            "seed":seed,"model":model,
            "phase2_core_accuracy":p2_acc,
            "final_third_factor_volition":TF,
            "final_ethical_reliability":R,
            "final_integration":I,
            "final_ideal_alignment":Align,
            "final_H_self":H,
            "phase2_reward_capture_rate":p2_capture,
            "developmental_transfer_score":transfer,
            "long_term_trajectory_value":traj/episodes
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v260.csv",index=False)

summary=df.groupby("model").agg(
    phase2_core_accuracy=("phase2_core_accuracy","mean"),
    final_third_factor_volition=("final_third_factor_volition","mean"),
    final_ethical_reliability=("final_ethical_reliability","mean"),
    final_integration=("final_integration","mean"),
    final_ideal_alignment=("final_ideal_alignment","mean"),
    final_H_self=("final_H_self","mean"),
    phase2_reward_capture_rate=("phase2_reward_capture_rate","mean"),
    developmental_transfer_score=("developmental_transfer_score","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean")
).reset_index()
summary.to_csv(BASE/"summary_v260.csv",index=False)

S=summary.set_index("model")
f="full_loop"
ae="ablate_ethicalsense"
ai="ablate_idealgap"
asv="ablate_shadow_validation"
ac="ablate_core_constraint"

checks={
    "full_beats_no_ethicalsense_transfer": bool(S.loc[f,"developmental_transfer_score"]>=S.loc[ae,"developmental_transfer_score"]+0.08),
    "full_beats_no_idealgap_transfer": bool(S.loc[f,"developmental_transfer_score"]>=S.loc[ai,"developmental_transfer_score"]+0.05),
    "full_beats_no_shadow_transfer": bool(S.loc[f,"developmental_transfer_score"]>=S.loc[asv,"developmental_transfer_score"]+0.03),
    "full_beats_no_core_transfer": bool(S.loc[f,"developmental_transfer_score"]>=S.loc[ac,"developmental_transfer_score"]+0.10),
    "core_ablation_increases_capture": bool(S.loc[ac,"phase2_reward_capture_rate"]>=S.loc[f,"phase2_reward_capture_rate"]+0.12),
    "ethicalsense_ablation_lowers_core_accuracy": bool(S.loc[ae,"phase2_core_accuracy"]<=S.loc[f,"phase2_core_accuracy"]-0.07),
    "shadow_ablation_increases_Hself_or_lowers_R": bool(
        S.loc[asv,"final_H_self"]>=S.loc[f,"final_H_self"]+0.05
        or S.loc[asv,"final_ethical_reliability"]<=S.loc[f,"final_ethical_reliability"]-0.05
    )
}
(BASE/"acceptance_v260.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
