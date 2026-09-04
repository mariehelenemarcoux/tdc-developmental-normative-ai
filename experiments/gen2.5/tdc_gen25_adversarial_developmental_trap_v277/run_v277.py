
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v277.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]
rows=[]; phase_rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]
        success_Q=Q

        total_traj=0.0
        trap_capture=0; trap_events=0
        false_consol=0; consol_attempts=0
        provisional_debt=0.0
        phase_metrics=[]

        for t in range(SPEC["episodes"]):
            pi=t//SPEC["phase_length"]
            phase=SPEC["phases"][pi]
            in_trap = phase["trap_strength"]>=0.60

            if t % SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0,"capture":0,"trap_events":0}

            # perception quality
            if model=="simple_time_schedule":
                frac=t/max(1,SPEC["episodes"]-1)
                q_use=0.58+(0.92-0.58)*frac
            elif model=="simple_success_driven":
                q_use=success_Q
            else:
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])
                q_use=Q

            critical=bool(rng.random()<E["critical_prob"])
            irreversible=bool(rng.random()<E["irreversible_given_critical"]) if critical else bool(rng.random()<0.12)
            correct_action=1 if critical else 0
            reward_action=correct_action if rng.random()<phase["reward_alignment"] else 1-correct_action

            norm_acc=clip(q_use-0.10*phase["trap_strength"],0.50,0.98)
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

            external_success = (action==reward_action)
            validated_success = bool(critical and irreversible and protected)
            normative_correct = (action==correct_action)
            normative_error = bool(critical and irreversible and not protected)

            if in_trap and reward_action!=correct_action:
                trap_events+=1; pst["trap_events"]+=1
                if action==reward_action:
                    trap_capture+=1; pst["capture"]+=1

            # model updates
            if model=="simple_success_driven":
                if external_success:
                    success_Q=clip(success_Q+0.007*(1-success_Q),0.50,0.94)
                else:
                    success_Q=clip(success_Q-0.002*success_Q,0.50,0.94)

            elif model in ("tdc_unvalidated","tdc_evidence_validated","tdc_validated_with_reversal"):
                candidate_signal = (
                    0.40*int(validated_success)
                    +0.30*int(external_success)
                    +0.20*(1-H)
                    +0.10*q_use
                )
                if rng.random()<phase["false_progress"]:
                    candidate_signal += 0.28

                accepted=True
                validation_score=1.0

                if model!="tdc_unvalidated":
                    validation_score = (
                        0.40*int(validated_success)
                        +0.30*int(normative_correct)
                        +0.20*int((not external_success) or reward_action==correct_action)
                        +0.10*q_use
                        + rng.normal(0,0.06)
                    )
                    accepted=validation_score>=D["validation_threshold"]

                if candidate_signal>0.35:
                    consol_attempts+=1
                    if accepted:
                        if normative_correct:
                            TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                            I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                            A=clip(A+D["eta_A"]*(1-A),0.20,0.95)
                            H=clip(H-D["eta_H"]*H,0.10,0.90)
                            if validated_success:
                                TF=clip(TF+D["validated_gain"]*q_use*(1-TF),0.20,0.95)
                                I=clip(I+0.8*D["validated_gain"]*q_use*(1-I),0.20,0.95)
                            if model=="tdc_validated_with_reversal":
                                provisional_debt=max(0.0,provisional_debt-0.02)
                        else:
                            false_consol+=1
                            I=clip(I-0.008,0.20,0.95)
                            A=clip(A-0.006,0.20,0.95)
                            H=clip(H+0.008,0.10,0.90)
                            if model=="tdc_validated_with_reversal":
                                provisional_debt+=0.06

                # reversal mechanism
                if model=="tdc_validated_with_reversal":
                    evidence_quality=0.55*validation_score+0.45*int(normative_correct)
                    if evidence_quality<D["reversal_trigger"] and provisional_debt>0:
                        reverse_amt=min(provisional_debt,D["reversal_rate"])
                        I=clip(I+0.5*reverse_amt*(0.34-I),0.20,0.95)
                        A=clip(A+0.5*reverse_amt*(0.40-A),0.20,0.95)
                        H=clip(H-0.3*reverse_amt*H,0.10,0.90)
                        provisional_debt=max(0.0,provisional_debt-reverse_amt)

                if normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                far=pst["fp"]/pst["n"]
                vr=pst["viol"]/max(1,pst["crit"])
                cap=pst["capture"]/max(1,pst["trap_events"])
                traj=pst["traj"]/pst["n"]
                phase_metrics.append((rec,far,vr,cap,traj,q_use))
                phase_rows.append({
                    "seed":seed,"model":model,"phase":phase["name"],
                    "recall":rec,"false_alarm":far,"violation":vr,
                    "trap_capture":cap,"trajectory":traj,"quality":q_use
                })
                total_traj += pst["traj"]

        final=phase_metrics[-1]
        trap_phase_cap=trap_capture/max(1,trap_events)
        false_consol_rate=false_consol/max(1,consol_attempts)
        post_trap_quality_damage=max(0.0,0.80-phase_metrics[3][5])
        recovery_gain=phase_metrics[4][0]-phase_metrics[3][0]

        integrated=(
            0.24*final[0]
            +0.16*(1-final[1])
            +0.18*(1-final[2])
            +0.14*(1-trap_phase_cap)
            +0.12*(1-false_consol_rate)
            +0.08*clip(recovery_gain+0.5,0,1)
            +0.08*clip((total_traj/SPEC["episodes"]+1)/2,0,1)
        )

        rows.append({
            "seed":seed,"model":model,
            "trap_phase_capture_rate":trap_phase_cap,
            "trap_phase_false_consolidation_rate":false_consol_rate,
            "post_trap_quality_damage":post_trap_quality_damage,
            "recovery_gain":recovery_gain,
            "final_holdout_recall":final[0],
            "final_holdout_false_alarm":final[1],
            "final_holdout_violation_rate":final[2],
            "trajectory_value":total_traj/SPEC["episodes"],
            "integrated_score":integrated,
            "final_quality":final[5]
        })

df=pd.DataFrame(rows); ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v277.csv",index=False)
ph.to_csv(BASE/"phase_results_v277.csv",index=False)

summary=df.groupby("model").agg(
    trap_phase_capture_rate=("trap_phase_capture_rate","mean"),
    trap_phase_false_consolidation_rate=("trap_phase_false_consolidation_rate","mean"),
    post_trap_quality_damage=("post_trap_quality_damage","mean"),
    recovery_gain=("recovery_gain","mean"),
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_false_alarm=("final_holdout_false_alarm","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    integrated_score=("integrated_score","mean"),
    final_quality=("final_quality","mean")
).reset_index()
summary.to_csv(BASE/"summary_v277.csv",index=False)

S=summary.set_index("model")
time="simple_time_schedule"; suc="simple_success_driven"
u="tdc_unvalidated"; v="tdc_evidence_validated"; r="tdc_validated_with_reversal"

strong_simple=max(S.loc[time,"integrated_score"],S.loc[suc,"integrated_score"])
best_tdc=max(S.loc[v,"integrated_score"],S.loc[r,"integrated_score"])

checks={
    "validated_lower_capture_than_time": bool(S.loc[v,"trap_phase_capture_rate"]<=S.loc[time,"trap_phase_capture_rate"]-0.10),
    "validated_lower_false_consolidation_than_unvalidated": bool(S.loc[v,"trap_phase_false_consolidation_rate"]<=S.loc[u,"trap_phase_false_consolidation_rate"]-0.12),
    "reversal_best_trap_resistance": bool(S.loc[r,"trap_phase_capture_rate"]<=S["trap_phase_capture_rate"].min()+1e-12),
    "reversal_recovers_better": bool(S.loc[r,"recovery_gain"]>=S.loc[v,"recovery_gain"]+0.03),
    "validated_better_final_recall_than_time": bool(S.loc[v,"final_holdout_recall"]>=S.loc[time,"final_holdout_recall"]+0.03),
    "validated_lower_final_violations_than_time": bool(S.loc[v,"final_holdout_violation_rate"]<=S.loc[time,"final_holdout_violation_rate"]-0.03),
    "complexity_earns_keep": bool(best_tdc>=strong_simple+0.03)
}
(BASE/"acceptance_v277.json").write_text(json.dumps({
    "checks":checks,
    "best_tdc_margin_vs_best_simple":float(best_tdc-strong_simple)
},indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,vv in checks.items(): print(f"{k}: {vv}")
print(f"passed={sum(checks.values())}/{len(checks)}")
print("Best TDC margin vs best simple:", best_tdc-strong_simple)
