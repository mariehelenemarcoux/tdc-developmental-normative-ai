
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v279.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]

        total_traj=0.0
        trap_capture=trap_events=0
        final_stats=None

        for t in range(SPEC["episodes"]):
            pi=t//SPEC["phase_length"]
            ph=SPEC["phases"][pi]
            if t % SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            if model=="simple_time_schedule":
                q_use=0.58+(0.92-0.58)*(t/(SPEC["episodes"]-1))
                anti=0.0
            elif model=="simple_anticapture":
                q_use=0.79
                anti=0.12
            elif model=="simple_frozen_core":
                q_use=0.80
                anti=0.18
            else:
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])
                q_use=Q
                anti=0.0

            critical=bool(rng.random()<E["critical_prob"])
            irreversible=bool(rng.random()<E["irreversible_given_critical"]) if critical else bool(rng.random()<0.12)
            correct_action=1 if critical else 0
            reward_action=correct_action if rng.random()<ph["reward_alignment"] else 1-correct_action

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal=noisy_bool(rng,critical,norm_acc)
            world_signal=noisy_bool(rng,critical,ph["world_acc"])
            irr_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_acc"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)

            # Simple anti-capture baselines explicitly discount reward-aligned pressure under traps.
            if model in ("simple_anticapture","simple_frozen_core") and reward_action==1:
                score-=anti*(1-ph["reward_alignment"])

            threshold=0.48 if irr_cue else 0.60

            if irr_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos=noisy_bool(rng,critical,probe_acc)
                score=clip(score+(0.20 if probe_pos else -0.16),0,1)
                pst["traj"]-=E["probe_cost"]

            action=1 if score>=threshold else 0
            trigger=(action==1)

            if trigger:
                if critical and irreversible: pst["tp"]+=1
                elif not critical: pst["fp"]+=1
            else:
                if critical and irreversible: pst["fn"]+=1

            protected=False
            if trigger:
                protected=bool(rng.random()<E["review_success"])
                pst["traj"]-=E["review_cost"]
                if not critical: pst["traj"]-=E["false_alarm_cost"]

            if critical and irreversible:
                pst["crit"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            if ph["trap"]>=0.50 and reward_action!=correct_action:
                trap_events+=1
                if action==reward_action: trap_capture+=1

            if model=="tdc_evidence_validated":
                validated_success=bool(critical and irreversible and protected)
                normative_correct=(action==correct_action)
                normative_error=bool(critical and irreversible and not protected)

                candidate=(
                    0.40*int(validated_success)
                    +0.28*int(action==reward_action)
                    +0.20*(1-H)
                    +0.12*q_use
                )
                if rng.random()<ph["false_progress"]:
                    candidate+=0.25

                validation=(
                    0.42*int(validated_success)
                    +0.30*int(normative_correct)
                    +0.18*int((action!=reward_action) or reward_action==correct_action)
                    +0.10*q_use
                    + rng.normal(0,0.06)
                )

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

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                far=pst["fp"]/pst["n"]
                vr=pst["viol"]/max(1,pst["crit"])
                traj=pst["traj"]/pst["n"]
                total_traj+=pst["traj"]
                if pi==len(SPEC["phases"])-1:
                    final_stats=(rec,far,vr,traj,q_use)

        final_rec,final_far,final_vr,final_phase_traj,final_q=final_stats
        capture=trap_capture/max(1,trap_events)
        trajectory=total_traj/SPEC["episodes"]

        integrated=(
            0.28*final_rec
            +0.18*(1-final_far)
            +0.20*(1-final_vr)
            +0.16*(1-capture)
            +0.10*clip((trajectory+1)/2,0,1)
            +0.08*final_q
        )

        rows.append({
            "seed":seed,"model":model,
            "trap_capture_rate":capture,
            "final_holdout_recall":final_rec,
            "final_holdout_false_alarm":final_far,
            "final_holdout_violation_rate":final_vr,
            "trajectory_value":trajectory,
            "integrated_score":integrated,
            "final_quality":final_q
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v279.csv",index=False)

summary=df.groupby("model").agg(
    trap_capture_rate=("trap_capture_rate","mean"),
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_false_alarm=("final_holdout_false_alarm","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    integrated_score=("integrated_score","mean"),
    final_quality=("final_quality","mean")
).reset_index()
summary.to_csv(BASE/"summary_v279.csv",index=False)

S=summary.set_index("model")
t="tdc_evidence_validated"
simple=["simple_time_schedule","simple_anticapture","simple_frozen_core"]
best_simple=S.loc[simple,"integrated_score"].max()
best_simple_name=S.loc[simple,"integrated_score"].idxmax()
lowest_simple_capture=S.loc[simple,"trap_capture_rate"].min()

checks={
    "tdc_best_recall": bool(S.loc[t,"final_holdout_recall"]>=S["final_holdout_recall"].max()-1e-12),
    "tdc_lowest_violations": bool(S.loc[t,"final_holdout_violation_rate"]<=S["final_holdout_violation_rate"].min()+1e-12),
    "tdc_lower_capture_than_best_simple": bool(S.loc[t,"trap_capture_rate"]<=lowest_simple_capture-0.03),
    "tdc_best_trajectory": bool(S.loc[t,"trajectory_value"]>=S["trajectory_value"].max()-1e-12),
    "tdc_best_integrated_score": bool(S.loc[t,"integrated_score"]>=S["integrated_score"].max()-1e-12),
    "positive_advantage_replication": bool(S.loc[t,"integrated_score"]>best_simple),
    "complexity_earns_keep": bool(S.loc[t,"integrated_score"]>=best_simple+0.03)
}
(BASE/"acceptance_v279.json").write_text(json.dumps({
    "checks":checks,
    "strongest_simple_baseline":best_simple_name,
    "tdc_margin_vs_strongest_simple":float(S.loc[t,"integrated_score"]-best_simple)
},indent=2))

print(summary.to_string(index=False))
print("\nStrongest simple baseline:",best_simple_name)
print("TDC margin:",S.loc[t,"integrated_score"]-best_simple)
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
