
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v287.json").read_text())
D=SPEC["development"]; L=SPEC["simple_learners"]; E=SPEC["environment"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))

for seed in SPEC["seeds"]:
    env_rng=np.random.default_rng(seed*1000+31)
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

    for mi,model in enumerate(SPEC["models"].keys()):
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]
        total_traj=0.0
        trap_capture=trap_events=0
        q_sum=0.0
        final_stats=None

        for t,e in enumerate(env):
            ph=SPEC["phases"][e["pi"]]
            if t%SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            if model=="tdc_full":
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                q_use=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                           D["quality_floor"],D["quality_ceiling"])
            else:
                q_use=Q

            q_sum+=q_use

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal = e["critical"] if e["r_core"]<norm_acc else (not e["critical"])
            world_signal = e["critical"] if e["r_world"]<ph["world_acc"] else (not e["critical"])
            irr_cue = e["irreversible"] if e["r_irr"]<E["irreversibility_cue_acc"] else (not e["irreversible"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)
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

            # updates
            if model=="tdc_full":
                candidate=(0.40*int(validated_success)+0.28*int(external_success)+0.20*(1-H)+0.12*q_use)
                if e["r_falseprog"]<ph["false_progress"]:
                    candidate+=0.25
                noise=(e["r_valnoise"]-0.5)*0.12
                validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                            +0.18*int((not external_success) or e["reward_action"]==e["correct_action"])
                            +0.10*q_use+noise)

                if candidate>0.35 and validation>=D["validation_threshold"] and normative_correct:
                    TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                    I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                    A=clip(A+D["eta_A"]*(1-A),0.20,0.95)
                    H=clip(H-D["eta_H"]*H,0.10,0.90)
                    if validated_success:
                        TF=clip(TF+D["validated_gain"]*q_use*(1-TF),0.20,0.95)
                        I=clip(I+0.8*D["validated_gain"]*q_use*(1-I),0.20,0.95)
                elif normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

            elif model=="simple_success_learner":
                if external_success:
                    Q=clip(Q+L["success_eta_up"]*(1-Q),L["q_min"],L["q_max"])
                else:
                    Q=clip(Q-L["success_eta_down"]*Q,L["q_min"],L["q_max"])

            elif model=="simple_calibrated_learner":
                # learns from audited normative correctness
                target=1.0 if normative_correct else 0.0
                Q=clip(Q+L["calibrated_eta"]*(target-Q),L["q_min"],L["q_max"])

            elif model=="simple_two_signal_learner":
                anti_capture = 1.0 if ((not external_success) or e["reward_action"]==e["correct_action"]) else 0.0
                target=0.65*int(normative_correct)+0.35*anti_capture
                Q=clip(Q+L["two_signal_eta"]*(target-Q),L["q_min"],L["q_max"])

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                vr=pst["viol"]/max(1,pst["crit"])
                total_traj+=pst["traj"]
                if e["pi"]==len(SPEC["phases"])-1:
                    final_stats=(rec,vr,q_use)

        final_rec,final_vr,final_q=final_stats
        capture=trap_capture/max(1,trap_events)

        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":final_rec,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "trajectory_value":total_traj/SPEC["episodes"],
            "final_Q":final_q,
            "mean_Q":q_sum/SPEC["episodes"]
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v287.csv",index=False)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    final_Q=("final_Q","mean"),
    mean_Q=("mean_Q","mean")
).reset_index()
summary.to_csv(BASE/"summary_v287.csv",index=False)

S=summary.set_index("model")
tdc="tdc_full"
simple=["simple_success_learner","simple_calibrated_learner","simple_two_signal_learner"]

best_recall=S.loc[simple,"final_holdout_recall"].max()
best_protection=(1-S.loc[simple,"final_holdout_violation_rate"]).max()
best_traj=S.loc[simple,"trajectory_value"].max()
best_capture_res=(1-S.loc[simple,"trap_capture_rate"]).max()
best_final_Q=S.loc[simple,"final_Q"].max()

# strongest simple by equal-weight behavior score
behavior_score={}
for m in simple:
    behavior_score[m]=(
        S.loc[m,"final_holdout_recall"]
        +(1-S.loc[m,"final_holdout_violation_rate"])
        +S.loc[m,"trajectory_value"]
        +(1-S.loc[m,"trap_capture_rate"])
    )/4
strongest=max(behavior_score,key=behavior_score.get)

wins=0
if S.loc[tdc,"final_holdout_recall"]>S.loc[strongest,"final_holdout_recall"]: wins+=1
if (1-S.loc[tdc,"final_holdout_violation_rate"])>(1-S.loc[strongest,"final_holdout_violation_rate"]): wins+=1
if S.loc[tdc,"trajectory_value"]>S.loc[strongest,"trajectory_value"]: wins+=1
if (1-S.loc[tdc,"trap_capture_rate"])>(1-S.loc[strongest,"trap_capture_rate"]): wins+=1

checks={
    "tdc_best_final_recall": bool(S.loc[tdc,"final_holdout_recall"]>=best_recall-1e-12),
    "tdc_lowest_violations": bool((1-S.loc[tdc,"final_holdout_violation_rate"])>=best_protection-1e-12),
    "tdc_best_trajectory": bool(S.loc[tdc,"trajectory_value"]>=best_traj-1e-12),
    "tdc_higher_final_Q_than_best_simple": bool(S.loc[tdc,"final_Q"]>=best_final_Q+0.03),
    "tdc_lower_capture_than_best_simple": bool((1-S.loc[tdc,"trap_capture_rate"])>=best_capture_res+0.03),
    "tdc_beats_best_simple_on_3_of_4_behavior_metrics": bool(wins>=3),
    "developmental_learning_adds_value": bool(wins>=3 and S.loc[tdc,"final_holdout_recall"]>=best_recall)
}

(BASE/"acceptance_v287.json").write_text(json.dumps({
    "checks":checks,
    "strongest_simple_learner":strongest,
    "tdc_wins_vs_strongest_simple":wins
},indent=2))

print(summary.to_string(index=False))
print("\nStrongest simple learner:",strongest)
print("TDC wins vs strongest simple:",wins,"/4")
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
