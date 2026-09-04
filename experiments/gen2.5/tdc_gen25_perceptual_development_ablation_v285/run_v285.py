
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v285.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]; M=SPEC["models"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,(model,cfg) in enumerate(M.items()):
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
            if t%SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            if cfg["type"]=="simple":
                q_use=cfg["quality"]; anti=cfg["anti"]
            else:
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])
                q_use=Q if cfg["type"]=="tdc_full" else SPEC["initial_state"]["Q"]
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
            if cfg["type"]=="simple" and reward_action==1:
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

            if ph["trap"]>=0.58 and reward_action!=correct_action:
                trap_events+=1
                if action==reward_action: trap_capture+=1

            if cfg["type"] in ("tdc_full","tdc_ablation"):
                validated_success=bool(critical and irreversible and protected)
                normative_correct=(action==correct_action)
                normative_error=bool(critical and irreversible and not protected)

                candidate=(0.40*int(validated_success)+0.28*int(action==reward_action)+0.20*(1-H)+0.12*q_use)
                if rng.random()<ph["false_progress"]:
                    candidate+=0.25

                validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                            +0.18*int((action!=reward_action) or reward_action==correct_action)
                            +0.10*q_use+rng.normal(0,0.06))

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
                total_traj+=pst["traj"]
                if pi==len(SPEC["phases"])-1:
                    final_stats=(rec,far,vr,q_use)

        final_rec,final_far,final_vr,final_q=final_stats
        capture=trap_capture/max(1,trap_events)
        maturity=(TF+I+A+(1-H))/4.0

        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":final_rec,
            "final_holdout_false_alarm":final_far,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "trajectory_value":total_traj/SPEC["episodes"],
            "final_quality_used":final_q,
            "final_TF":TF,"final_I":I,"final_A":A,"final_H":H,
            "final_maturity":maturity
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v285.csv",index=False)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_false_alarm=("final_holdout_false_alarm","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    final_quality_used=("final_quality_used","mean"),
    final_TF=("final_TF","mean"),
    final_I=("final_I","mean"),
    final_A=("final_A","mean"),
    final_H=("final_H","mean"),
    final_maturity=("final_maturity","mean")
).reset_index()
summary.to_csv(BASE/"summary_v285.csv",index=False)

S=summary.set_index("model")
full="tdc_full_developmental_perception"
abl="tdc_no_perceptual_development"
simp="simple_fixed_q058"

initial_maturity=(0.24+0.34+0.40+(1-0.62))/4.0
full_minus_abl_recall=S.loc[full,"final_holdout_recall"]-S.loc[abl,"final_holdout_recall"]
abl_minus_simple_recall=S.loc[abl,"final_holdout_recall"]-S.loc[simp,"final_holdout_recall"]
full_minus_abl_traj=S.loc[full,"trajectory_value"]-S.loc[abl,"trajectory_value"]
abl_minus_simple_traj=S.loc[abl,"trajectory_value"]-S.loc[simp,"trajectory_value"]

checks={
    "ablation_reduces_recall": bool(S.loc[abl,"final_holdout_recall"]<=S.loc[full,"final_holdout_recall"]-0.10),
    "ablation_increases_violations": bool(S.loc[abl,"final_holdout_violation_rate"]>=S.loc[full,"final_holdout_violation_rate"]+0.08),
    "ablation_worse_trajectory": bool(S.loc[abl,"trajectory_value"]<=S.loc[full,"trajectory_value"]-0.08),
    "internal_development_still_occurs": bool(S.loc[abl,"final_maturity"]>=initial_maturity+0.20),
    "ablated_tdc_beats_simple_q058_recall": bool(S.loc[abl,"final_holdout_recall"]>=S.loc[simp,"final_holdout_recall"]+0.05),
    "ablated_tdc_beats_simple_q058_trajectory": bool(S.loc[abl,"trajectory_value"]>=S.loc[simp,"trajectory_value"]+0.05),
    "perception_is_primary_advantage_channel": bool(
        full_minus_abl_recall>=0.10 and full_minus_abl_traj>=0.08
        and abl_minus_simple_recall<0.05 and abl_minus_simple_traj<0.05
    )
}

(BASE/"acceptance_v285.json").write_text(json.dumps({
    "checks":checks,
    "initial_maturity":initial_maturity,
    "full_minus_ablation_recall":float(full_minus_abl_recall),
    "ablation_minus_simple_recall":float(abl_minus_simple_recall),
    "full_minus_ablation_trajectory":float(full_minus_abl_traj),
    "ablation_minus_simple_trajectory":float(abl_minus_simple_traj)
},indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
