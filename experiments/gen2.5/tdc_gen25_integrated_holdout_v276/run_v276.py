
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v276.json").read_text())
D=SPEC["development"]; B=SPEC["simple_baselines"]; E=SPEC["environment"]
rows=[]; phase_rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]; Qonline=B["online_initial"]
        total_traj=0.0
        total_capture=0; total_deceptive=0
        false_consol=0; consol_attempts=0
        phase_metrics=[]

        for t in range(SPEC["episodes"]):
            pi=t//SPEC["phase_length"]
            phase=SPEC["phases"][pi]
            if t % SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0,"capture":0,"deceptive":0}

            # model quality
            if model=="simple_time_schedule":
                frac=t/max(1,SPEC["episodes"]-1)
                q_use=B["schedule_start"]+(B["schedule_end"]-B["schedule_start"])*frac
            elif model=="simple_online_calibration":
                q_use=Qonline
            else:
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,D["quality_floor"],D["quality_ceiling"])
                q_use=Q

            critical=bool(rng.random()<E["critical_prob"])
            irreversible=bool(rng.random()<E["irreversible_given_critical"]) if critical else bool(rng.random()<0.12)

            # latent correct ethical action: block if critical
            correct_action = 1 if critical else 0
            reward_action = correct_action if rng.random()<phase["reward_alignment"] else 1-correct_action

            # noisy signals
            norm_acc=clip(q_use-phase["normative_noise"],0.50,0.98)
            core_signal=noisy_bool(rng,critical,norm_acc)
            world_signal=noisy_bool(rng,critical,phase["world_acc"])
            irreversible_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_acc"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irreversible_cue)
            threshold=0.48 if irreversible_cue else 0.60

            if irreversible_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos=noisy_bool(rng,critical,probe_acc)
                score=clip(score+(0.20 if probe_pos else -0.16),0,1)
                pst["traj"]-=E["probe_cost"]

            action=1 if score>=threshold else 0
            trigger=action==1

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

            # deceptive capture metric: action follows reward when reward is normatively wrong
            if reward_action != correct_action:
                pst["deceptive"]+=1
                total_deceptive+=1
                if action==reward_action:
                    pst["capture"]+=1
                    total_capture+=1

            validated_success=bool(critical and irreversible and protected)
            normative_error=bool(critical and irreversible and not protected)

            if model=="simple_online_calibration":
                if validated_success:
                    Qonline=clip(Qonline+B["online_up"]*(1-Qonline),0.50,0.94)
                elif normative_error:
                    Qonline=clip(Qonline-B["online_down"]*Qonline,0.50,0.94)

            elif model in ("tdc_feedback_unvalidated","tdc_evidence_validated"):
                # candidate developmental event
                candidate_signal = (
                    0.45*int(validated_success)
                    +0.25*int(action==reward_action)
                    +0.20*(1-H)
                    +0.10*q_use
                )
                # false progress lure
                if rng.random()<phase["false_progress"]:
                    candidate_signal += 0.25

                should_validate = model=="tdc_evidence_validated"
                accepted=True

                if should_validate:
                    # independent noisy post-action validation: core consistency + outcome + anti-capture
                    validation = (
                        0.45*int(validated_success)
                        +0.25*int(action==correct_action)
                        +0.20*int(action!=reward_action or reward_action==correct_action)
                        +0.10*q_use
                    )
                    validation += rng.normal(0,0.06)
                    accepted = validation>=D["validation_threshold"]

                if candidate_signal>0.35:
                    consol_attempts+=1
                    if accepted:
                        if action==correct_action:
                            TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                            I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                            A=clip(A+D["eta_A"]*(1-A),0.20,0.95)
                            H=clip(H-D["eta_H"]*H,0.10,0.90)
                            if validated_success:
                                TF=clip(TF+D["validated_gain"]*q_use*(1-TF),0.20,0.95)
                                I=clip(I+0.8*D["validated_gain"]*q_use*(1-I),0.20,0.95)
                        else:
                            false_consol+=1
                            # unvalidated false consolidation degrades structure
                            I=clip(I-0.005,0.20,0.95)
                            A=clip(A-0.004,0.20,0.95)
                            H=clip(H+0.006,0.10,0.90)

                if normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                far=pst["fp"]/pst["n"]
                vr=pst["viol"]/max(1,pst["crit"])
                cap=pst["capture"]/max(1,pst["deceptive"])
                traj=pst["traj"]/pst["n"]
                phase_metrics.append((rec,far,vr,cap,traj,q_use))
                phase_rows.append({
                    "seed":seed,"model":model,"phase":phase["name"],
                    "recall":rec,"false_alarm":far,"violation":vr,
                    "capture":cap,"trajectory":traj,"quality":q_use
                })
                total_traj += pst["traj"]

        final=phase_metrics[-1]
        recovery=phase_metrics[4][0]-phase_metrics[3][0]
        final_recall,final_far,final_vr,final_cap,final_traj,final_q=final
        capture_rate=total_capture/max(1,total_deceptive)
        false_consol_rate=false_consol/max(1,consol_attempts)

        integrated=(
            0.28*final_recall
            +0.16*(1-final_far)
            +0.18*(1-final_vr)
            +0.14*(1-capture_rate)
            +0.10*(1-false_consol_rate)
            +0.08*clip(recovery+0.5,0,1)
            +0.06*clip((total_traj/SPEC["episodes"]+1)/2,0,1)
        )

        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":final_recall,
            "final_holdout_false_alarm":final_far,
            "final_holdout_violation_rate":final_vr,
            "trajectory_value":total_traj/SPEC["episodes"],
            "capture_rate_deceptive_phases":capture_rate,
            "false_consolidation_rate":false_consol_rate,
            "recovery_gain":recovery,
            "net_integrated_score":integrated,
            "final_quality":final_q
        })

df=pd.DataFrame(rows); ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v276.csv",index=False)
ph.to_csv(BASE/"phase_results_v276.csv",index=False)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_false_alarm=("final_holdout_false_alarm","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    capture_rate_deceptive_phases=("capture_rate_deceptive_phases","mean"),
    false_consolidation_rate=("false_consolidation_rate","mean"),
    recovery_gain=("recovery_gain","mean"),
    net_integrated_score=("net_integrated_score","mean"),
    final_quality=("final_quality","mean")
).reset_index()
summary.to_csv(BASE/"summary_v276.csv",index=False)

S=summary.set_index("model")
t="tdc_evidence_validated"; u="tdc_feedback_unvalidated"
sc="simple_time_schedule"; on="simple_online_calibration"
simple_best=S.loc[[sc,on],"net_integrated_score"].max()
all_best=S["net_integrated_score"].max()

checks={
    "tdc_validated_best_holdout_recall": bool(S.loc[t,"final_holdout_recall"]>=S["final_holdout_recall"].max()-1e-12),
    "tdc_validated_lower_capture_than_schedule": bool(S.loc[t,"capture_rate_deceptive_phases"]<=S.loc[sc,"capture_rate_deceptive_phases"]-0.08),
    "tdc_validated_lower_false_consolidation": bool(S.loc[t,"false_consolidation_rate"]<=S.loc[u,"false_consolidation_rate"]-0.10),
    "tdc_validated_lower_violations_than_schedule": bool(S.loc[t,"final_holdout_violation_rate"]<=S.loc[sc,"final_holdout_violation_rate"]-0.03),
    "tdc_validated_better_trajectory_than_schedule": bool(S.loc[t,"trajectory_value"]>=S.loc[sc,"trajectory_value"]+0.03),
    "tdc_validated_best_integrated_score": bool(S.loc[t,"net_integrated_score"]>=all_best-1e-12),
    "complexity_earns_keep": bool(S.loc[t,"net_integrated_score"]>=simple_best+0.03)
}
(BASE/"acceptance_v276.json").write_text(json.dumps({
    "checks":checks,
    "strongest_simple_score":float(simple_best),
    "tdc_margin_vs_strongest_simple":float(S.loc[t,"net_integrated_score"]-simple_best)
},indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
print("TDC margin vs strongest simple:", S.loc[t,"net_integrated_score"]-simple_best)
