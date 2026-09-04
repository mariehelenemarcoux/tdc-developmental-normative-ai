
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v286.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    # Pre-generate common environment randomness to make the comparison paired.
    env_rng=np.random.default_rng(seed*1000+7)
    env=[]
    for t in range(SPEC["episodes"]):
        pi=t//SPEC["phase_length"]; ph=SPEC["phases"][pi]
        critical=bool(env_rng.random()<E["critical_prob"])
        irreversible=bool(env_rng.random()<E["irreversible_given_critical"]) if critical else bool(env_rng.random()<0.12)
        correct_action=1 if critical else 0
        reward_action=correct_action if env_rng.random()<ph["reward_alignment"] else 1-correct_action
        env.append((pi,critical,irreversible,correct_action,reward_action,
                    env_rng.random(),env_rng.random(),env_rng.random(),env_rng.random(),
                    env_rng.random(),env_rng.random(),env_rng.random()))

    # First pass: full TDC generates Q trajectory and its own outcomes.
    TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
    A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
    q_traj=[]
    model_results={}

    for model in ["tdc_full","simple_matched_Q","simple_matched_Q_anticapture","simple_static_q084"]:
        rng=np.random.default_rng(seed*100+{"tdc_full":0,"simple_matched_Q":1,"simple_matched_Q_anticapture":2,"simple_static_q084":3}[model])

        total_traj=0.0; trap_capture=trap_events=0; final_stats=None
        if model=="tdc_full":
            TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
            A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
            q_traj=[]

        for t in range(SPEC["episodes"]):
            pi,critical,irreversible,correct_action,reward_action,r_core,r_world,r_irr,r_probe,r_review,r_fp1,r_fp2=env[t]
            ph=SPEC["phases"][pi]
            if t%SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            if model=="tdc_full":
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                q_use=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                           D["quality_floor"],D["quality_ceiling"])
                q_traj.append(q_use)
                anti=0.0
            elif model=="simple_matched_Q":
                q_use=q_traj[t]; anti=0.0
            elif model=="simple_matched_Q_anticapture":
                q_use=q_traj[t]; anti=0.08
            else:
                q_use=0.84; anti=0.0

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal = critical if r_core<norm_acc else (not critical)
            world_signal = critical if r_world<ph["world_acc"] else (not critical)
            irr_cue = irreversible if r_irr<E["irreversibility_cue_acc"] else (not irreversible)

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)
            if anti>0 and reward_action==1:
                score-=anti*(1-ph["reward_alignment"])

            threshold=0.48 if irr_cue else 0.60

            if irr_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos = critical if r_probe<probe_acc else (not critical)
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
                protected=(r_review<E["review_success"])
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

            if model=="tdc_full":
                validated_success=bool(critical and irreversible and protected)
                normative_correct=(action==correct_action)
                normative_error=bool(critical and irreversible and not protected)

                candidate=(0.40*int(validated_success)+0.28*int(action==reward_action)+0.20*(1-H)+0.12*q_use)
                if r_fp1<ph["false_progress"]:
                    candidate+=0.25

                # paired but noisy validation
                noise=(r_fp2-0.5)*0.12
                validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                            +0.18*int((action!=reward_action) or reward_action==correct_action)
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

            pst["n"]+=1
            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                vr=pst["viol"]/max(1,pst["crit"])
                total_traj+=pst["traj"]
                if pi==len(SPEC["phases"])-1:
                    final_stats=(rec,vr,q_use)

        final_rec,final_vr,final_q=final_stats
        model_results[model]={
            "recall":final_rec,
            "violation":final_vr,
            "capture":trap_capture/max(1,trap_events),
            "trajectory":total_traj/SPEC["episodes"],
            "mean_Q":float(np.mean(q_traj)) if "matched_Q" in model or model=="tdc_full" else 0.84,
            "final_Q":final_q
        }

    for model,res in model_results.items():
        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":res["recall"],
            "final_holdout_violation_rate":res["violation"],
            "trap_capture_rate":res["capture"],
            "trajectory_value":res["trajectory"],
            "mean_Q":res["mean_Q"],
            "final_Q":res["final_Q"]
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v286.csv",index=False)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    mean_Q=("mean_Q","mean"),
    final_Q=("final_Q","mean")
).reset_index()
summary.to_csv(BASE/"summary_v286.csv",index=False)

S=summary.set_index("model")
t="tdc_full"; m="simple_matched_Q"; ma="simple_matched_Q_anticapture"

diffs={
    "recall":abs(S.loc[t,"final_holdout_recall"]-S.loc[m,"final_holdout_recall"]),
    "violations":abs(S.loc[t,"final_holdout_violation_rate"]-S.loc[m,"final_holdout_violation_rate"]),
    "trajectory":abs(S.loc[t,"trajectory_value"]-S.loc[m,"trajectory_value"]),
    "capture":abs(S.loc[t,"trap_capture_rate"]-S.loc[m,"trap_capture_rate"])
}
close_count=sum([
    diffs["recall"]<=0.02,
    diffs["violations"]<=0.02,
    diffs["trajectory"]<=0.02,
    diffs["capture"]<=0.03
])

tdc_adv_metrics=0
if S.loc[t,"final_holdout_recall"]>=S.loc[m,"final_holdout_recall"]+0.03: tdc_adv_metrics+=1
if (1-S.loc[t,"final_holdout_violation_rate"]) >= (1-S.loc[m,"final_holdout_violation_rate"])+0.03: tdc_adv_metrics+=1
if S.loc[t,"trajectory_value"]>=S.loc[m,"trajectory_value"]+0.03: tdc_adv_metrics+=1
if (1-S.loc[t,"trap_capture_rate"]) >= (1-S.loc[m,"trap_capture_rate"])+0.03: tdc_adv_metrics+=1

checks={
    "matched_Q_recall_within_0_02": bool(diffs["recall"]<=0.02),
    "matched_Q_violations_within_0_02": bool(diffs["violations"]<=0.02),
    "matched_Q_trajectory_within_0_02": bool(diffs["trajectory"]<=0.02),
    "matched_Q_capture_within_0_03": bool(diffs["capture"]<=0.03),
    "anticapture_matched_Q_not_worse_than_tdc_on_capture": bool(S.loc[ma,"trap_capture_rate"]<=S.loc[t,"trap_capture_rate"]),
    "tdc_has_residual_architectural_advantage": bool(tdc_adv_metrics>=2),
    "perception_explains_most_advantage": bool(close_count>=3)
}

(BASE/"acceptance_v286.json").write_text(json.dumps({
    "checks":checks,
    "absolute_differences_tdc_vs_matched_Q":{k:float(v) for k,v in diffs.items()},
    "close_metric_count":int(close_count),
    "tdc_advantage_metric_count":int(tdc_adv_metrics)
},indent=2))

print(summary.to_string(index=False))
print("\nDifferences TDC vs matched-Q:",diffs)
print("Close metrics:",close_count,"/4")
print("TDC residual advantage metrics >=.03:",tdc_adv_metrics)
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
