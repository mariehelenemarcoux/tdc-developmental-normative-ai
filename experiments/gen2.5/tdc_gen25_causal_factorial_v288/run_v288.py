
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v288.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]; MODELS=SPEC["models"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))

for seed in SPEC["seeds"]:
    env_rng=np.random.default_rng(seed*1000+77)
    env=[]
    for t in range(SPEC["episodes"]):
        pi=t//SPEC["phase_length"]; ph=SPEC["phases"][pi]
        critical=bool(env_rng.random()<E["critical_prob"])
        irreversible=bool(env_rng.random()<E["irreversible_given_critical"]) if critical else bool(env_rng.random()<0.12)
        correct_action=1 if critical else 0
        reward_action=correct_action if env_rng.random()<ph["reward_alignment"] else 1-correct_action
        env.append({
            "pi":pi,"critical":critical,"irreversible":irreversible,
            "correct_action":correct_action,"reward_action":reward_action,
            "r_core":env_rng.random(),"r_world":env_rng.random(),"r_irr":env_rng.random(),
            "r_probe":env_rng.random(),"r_review":env_rng.random(),
            "r_falseprog":env_rng.random(),"r_valnoise":env_rng.random()
        })

    for model, flags in MODELS.items():
        P,T,V,M,Aanti = flags
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        Align=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]
        mem=0.0

        total_traj=0.0
        trap_capture=trap_events=0
        final_stats=None

        for t,e in enumerate(env):
            ph=SPEC["phases"][e["pi"]]
            if t%SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            q_use=Q

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal=e["critical"] if e["r_core"]<norm_acc else (not e["critical"])
            world_signal=e["critical"] if e["r_world"]<ph["world_acc"] else (not e["critical"])
            irr_cue=e["irreversible"] if e["r_irr"]<E["irreversibility_cue_acc"] else (not e["irreversible"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)

            if M:
                score += 0.10*mem

            if Aanti and e["reward_action"]==1:
                score -= D["anti_capture_strength"]*(1-ph["reward_alignment"])

            if T:
                # Third Factor as a small authority bias toward internal normative evidence under conflict.
                conflict = (e["reward_action"] != int(core_signal))
                if conflict:
                    score += 0.06*(2*int(core_signal)-1)*TF

            threshold=0.48 if irr_cue else 0.60

            if irr_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos=e["critical"] if e["r_probe"]<probe_acc else (not e["critical"])
                score=clip(score+(0.20 if probe_pos else -0.16),0,1)
                pst["traj"]-=E["probe_cost"]

            action=1 if score>=threshold else 0
            trigger=(action==1)

            if trigger:
                if e["critical"] and e["irreversible"]: pst["tp"]+=1
                elif not e["critical"]: pst["fp"]+=1
            else:
                if e["critical"] and e["irreversible"]: pst["fn"]+=1

            protected=False
            if trigger:
                protected=e["r_review"]<E["review_success"]
                pst["traj"]-=E["review_cost"]
                if not e["critical"]:
                    pst["traj"]-=E["false_alarm_cost"]

            if e["critical"] and e["irreversible"]:
                pst["crit"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            normative_correct=(action==e["correct_action"])
            external_success=(action==e["reward_action"])
            validated_success=bool(e["critical"] and e["irreversible"] and protected)
            normative_error=bool(e["critical"] and e["irreversible"] and not protected)

            if ph["trap"]>=0.60 and e["reward_action"]!=e["correct_action"]:
                trap_events+=1
                if action==e["reward_action"]:
                    trap_capture+=1

            # Memory stores correctness evidence, with decay.
            if M:
                mem=D["memory_decay"]*mem + D["memory_gain"]*((1 if normative_correct else -1))
                mem=clip(mem,-1,1)

            # Validation gate
            accepted=True
            if V:
                noise=(e["r_valnoise"]-0.5)*0.12
                validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                            +0.18*int((not external_success) or e["reward_action"]==e["correct_action"])
                            +0.10*q_use+noise)
                accepted=validation>=D["validation_threshold"]

            # Perceptual learning
            if P:
                if (not V) or accepted:
                    target=1.0 if normative_correct else 0.0
                    # with Third Factor, learning is weighted toward normatively coherent updates
                    gain=D["eta_Q"]*(1.15 if (T and normative_correct) else 1.0)
                    Q=clip(Q+gain*(target-Q),0.50,0.94)

            # Internal developmental state
            if T:
                if validated_success and ((not V) or accepted):
                    TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                    I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                    Align=clip(Align+D["eta_A"]*(1-Align),0.20,0.95)
                    H=clip(H-D["eta_H"]*H,0.10,0.90)
                elif normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                vr=pst["viol"]/max(1,pst["crit"])
                total_traj+=pst["traj"]
                if e["pi"]==len(SPEC["phases"])-1:
                    final_stats=(rec,vr,Q)

        final_rec,final_vr,final_Q=final_stats
        capture=trap_capture/max(1,trap_events)

        rows.append({
            "seed":seed,"model":model,
            "P":P,"T":T,"V":V,"M":M,"Aanti":Aanti,
            "final_holdout_recall":final_rec,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "trajectory_value":total_traj/SPEC["episodes"],
            "final_Q":final_Q
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v288.csv",index=False)

df["composite"]=(
    0.30*df["final_holdout_recall"]
    +0.25*(1-df["final_holdout_violation_rate"])
    +0.20*(1-df["trap_capture_rate"])
    +0.25*df["trajectory_value"]
)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    final_Q=("final_Q","mean"),
    composite=("composite","mean")
).reset_index()
summary.to_csv(BASE/"summary_v288.csv",index=False)

# Main effects: average score when factor on minus average when off across preregistered model set.
effects={}
for col in ["P","T","V","M","Aanti"]:
    on=df[df[col]==1]["composite"].mean()
    off=df[df[col]==0]["composite"].mean()
    effects[col]=float(on-off)

effects_df=pd.DataFrame([{"factor":k,"main_effect":v} for k,v in effects.items()])
effects_df.to_csv(BASE/"main_effects_v288.csv",index=False)

S=summary.set_index("model")
best_reduced=summary[summary["model"]!="FULL"]["composite"].max()
largest=max(effects,key=lambda k: effects[k])

checks={
    "perceptual_learning_positive_main_effect": bool(effects["P"]>0.04),
    "third_factor_positive_main_effect": bool(effects["T"]>0),
    "validation_positive_main_effect": bool(effects["V"]>0),
    "memory_positive_main_effect": bool(effects["M"]>0),
    "anti_capture_positive_main_effect": bool(effects["Aanti"]>0),
    "perception_largest_main_effect": bool(largest=="P"),
    "full_not_worse_than_best_reduced": bool(S.loc["FULL","composite"]>=best_reduced-0.01)
}

(BASE/"acceptance_v288.json").write_text(json.dumps({
    "checks":checks,
    "main_effects":effects,
    "largest_main_effect":largest,
    "best_reduced_composite":float(best_reduced),
    "full_composite":float(S.loc["FULL","composite"])
},indent=2))

print(summary.to_string(index=False))
print("\nMAIN EFFECTS")
print(effects_df.sort_values("main_effect",ascending=False).to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
