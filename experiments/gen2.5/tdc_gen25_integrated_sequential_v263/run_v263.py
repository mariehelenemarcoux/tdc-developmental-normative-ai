
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v263.json").read_text())
E=SPEC["environment"]
P=SPEC["developmental_parameters"]
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

        stats={1:{"core_ok":0,"n":0},2:{"core_ok":0,"n":0}}
        reward_capture_p2=0
        false_consolidations=0
        accepted_candidates=0
        total_candidates=0
        traj=0.0

        for t in range(episodes):
            phase=1 if t<p1n else 2
            reward_align_prob=E["phase1_reward_alignment_probability"] if phase==1 else E["phase2_reward_alignment_probability"]

            fact=1 if rng.random()<0.5 else -1
            core=1 if rng.random()<0.5 else -1
            long=1 if rng.random()<0.5 else -1

            latent_score=0.46*fact+0.90*core+0.78*long+0.44*core*long
            y=sign(latent_score)

            fact_obs=noisy(rng,fact,E["world_signal_accuracy"])
            core_obs=noisy(rng,core,E["normative_signal_accuracy"])
            long_obs=noisy(rng,long,E["normative_signal_accuracy"])

            high=bool(rng.random()<E["high_stakes_probability"])
            pressure=bool(rng.random()<E["external_pressure_probability"])
            pressure_action=(-y if rng.random()<E["pressure_wrong_probability"] else y) if pressure else None

            reward_action=y if rng.random()<reward_align_prob else -y

            world_est=fact_obs if rng.random()<W else -fact_obs
            norm_ground=sign(0.60*core+0.40*long)
            norm_obs=sign(0.60*core_obs+0.40*long_obs)
            ethical_interp=norm_obs if rng.random()<R else -norm_obs
            internal_action=sign(0.34*world_est+0.66*ethical_interp)

            conflict=bool(pressure and pressure_action!=internal_action)
            core_long_disagree=(core_obs!=long_obs)

            if model=="ablate_ethicalsense":
                ethical_salience=clip(
                    0.40*(1 if high else 0)+0.30*(1 if pressure else 0)+0.30*(1 if fact_obs!=world_est else 0),
                    0,1
                )
            else:
                ethical_salience=clip(
                    0.32*(1 if conflict else 0)
                    +0.30*(1 if high else 0)
                    +0.38*(1 if core_long_disagree else 0),0,1
                )

            ideal_gap=max(0.0,1.0-Align)
            if model=="ablate_idealgap":
                drive=ethical_salience*R
            else:
                drive=ethical_salience*R*ideal_gap

            authority=clip(0.42*TF+0.22*R+0.16*I+0.12*Align+0.08*(1-H),0,1)
            independent=authority>=0.30
            action=pressure_action if (pressure and not independent) else internal_action

            ok=(action==y)
            stats[phase]["core_ok"]+=int(ok)
            stats[phase]["n"]+=1

            if phase==2 and action==reward_action and reward_action!=y:
                reward_capture_p2+=1

            if ok:
                traj+=1.0
            elif high:
                traj-=E["severe_error_penalty"]
            if independent and high:
                traj-=E["deep_cost"]

            normative_correct=(ethical_interp==norm_ground)

            if drive>P["candidate_threshold"]:
                total_candidates+=1

                # candidate may be seductive and wrong
                seductive_wrong = bool(rng.random()<E["seductive_wrong_candidate_probability"])
                if seductive_wrong:
                    candidate_correct=False
                else:
                    candidate_correct = rng.random() < clip(0.52+0.26*R+0.18*I,0.50,0.95)

                if model=="ablate_core_constraint":
                    # reward can redefine direction: if reward disagrees with core, candidate is more likely reward-aligned
                    if reward_action!=y and rng.random()<0.70:
                        candidate_correct=False

                if model=="ablate_shadow_validation":
                    accepted=True
                else:
                    accepted = candidate_correct if rng.random()<E["shadow_validation_accuracy"] else (not candidate_correct)

                if accepted:
                    accepted_candidates+=1
                    if not candidate_correct:
                        false_consolidations+=1

                    TF=clip(TF+P["eta_TF"]*drive,init["third_factor_volition"],0.95)

                    if candidate_correct:
                        I=clip(I+P["eta_I"]*drive,0.10,0.95)

                        if model=="ablate_core_constraint" and reward_action!=y:
                            Align=clip(Align-P["eta_align"]*0.55*drive,0.10,0.95)
                        else:
                            Align=clip(Align+P["eta_align"]*drive,0.10,0.95)

                        R=clip(R+P["eta_R"]*I*drive,0.50,0.94)
                        H=clip(H-P["eta_H_down"]*drive,0.10,0.90)
                    else:
                        H=clip(H+P["eta_H_up"]*drive,0.10,0.90)
                        R=clip(R-0.002*drive,0.50,0.94)

            if ethical_salience>0.35 and model!="ablate_ethicalsense":
                R=clip(R+(0.0013*I if normative_correct else -0.0011),0.50,0.94)

        p2_acc=stats[2]["core_ok"]/stats[2]["n"]
        p2_cap=reward_capture_p2/max(1,stats[2]["n"])
        false_rate=false_consolidations/max(1,accepted_candidates)

        integrated=(
            0.25*p2_acc
            +0.20*Align
            +0.15*I
            +0.15*(1-H)
            +0.15*(1-p2_cap)
            +0.10*(1-false_rate)
        )

        rows.append({
            "seed":seed,"model":model,
            "phase2_core_accuracy":p2_acc,
            "final_third_factor_volition":TF,
            "final_ethical_reliability":R,
            "final_integration":I,
            "final_ideal_alignment":Align,
            "final_H_self":H,
            "phase2_reward_capture_rate":p2_cap,
            "false_consolidation_rate":false_rate,
            "integrated_score":integrated,
            "long_term_trajectory_value":traj/episodes
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v263.csv",index=False)

summary=df.groupby("model").agg(
    phase2_core_accuracy=("phase2_core_accuracy","mean"),
    final_third_factor_volition=("final_third_factor_volition","mean"),
    final_ethical_reliability=("final_ethical_reliability","mean"),
    final_integration=("final_integration","mean"),
    final_ideal_alignment=("final_ideal_alignment","mean"),
    final_H_self=("final_H_self","mean"),
    phase2_reward_capture_rate=("phase2_reward_capture_rate","mean"),
    false_consolidation_rate=("false_consolidation_rate","mean"),
    integrated_score=("integrated_score","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean")
).reset_index()
summary.to_csv(BASE/"summary_v263.csv",index=False)

S=summary.set_index("model")
f="full_loop"
abl=["ablate_ethicalsense","ablate_idealgap","ablate_shadow_validation","ablate_core_constraint"]

mean_acc=S.loc[abl,"phase2_core_accuracy"].mean()
mean_align=S.loc[abl,"final_ideal_alignment"].mean()
best_traj=S.loc[abl,"long_term_trajectory_value"].max()

targeted_hurt = (
    S.loc["ablate_ethicalsense","phase2_core_accuracy"] <= S.loc[f,"phase2_core_accuracy"]-0.04
    and S.loc["ablate_idealgap","final_ideal_alignment"] <= S.loc[f,"final_ideal_alignment"]-0.03
    and S.loc["ablate_shadow_validation","false_consolidation_rate"] >= S.loc[f,"false_consolidation_rate"]+0.12
    and S.loc["ablate_core_constraint","phase2_reward_capture_rate"] >= S.loc[f,"phase2_reward_capture_rate"]+0.08
)

checks={
    "full_best_integrated_score": bool(all(S.loc[f,"integrated_score"]>=S.loc[m,"integrated_score"]+0.04 for m in abl)),
    "full_core_accuracy_advantage": bool(S.loc[f,"phase2_core_accuracy"]>=mean_acc+0.05),
    "full_low_capture": bool(S.loc[f,"phase2_reward_capture_rate"]<=0.24),
    "full_low_false_consolidation": bool(S.loc[f,"false_consolidation_rate"]<=S.loc["ablate_shadow_validation","false_consolidation_rate"]-0.12),
    "full_alignment_advantage": bool(S.loc[f,"final_ideal_alignment"]>=mean_align+0.05),
    "full_longterm_not_worse": bool(S.loc[f,"long_term_trajectory_value"]>=best_traj-0.02),
    "all_targeted_ablations_hurt": bool(targeted_hurt)
}
(BASE/"acceptance_v263.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
