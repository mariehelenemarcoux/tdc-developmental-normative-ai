
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v270.json").read_text())
E=SPEC["environment"]
D=SPEC["detectors"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)

        tp_i=fn_i=0
        tp_r=fn_r=0
        fp=0
        reviews=0
        irreversible_viol=0
        reversible_miss=0
        traj=0.0

        if model=="fixed_threshold":
            th= D["fixed_threshold"]["threshold"]
        elif model=="global_recall_constrained":
            th=D[model]["initial_threshold"]
        elif model=="hierarchical_recall_constrained":
            th_i=D[model]["initial_threshold_irreversible"]
            th_r=D[model]["initial_threshold_reversible"]
        else:
            th_i=D[model]["threshold_irreversible"]
            th_r=D[model]["threshold_reversible"]

        # running audit estimates
        rtp_i=rfn_i=2
        rtp_r=rfn_r=2

        for t in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            if critical:
                irreversible=bool(rng.random()<E["irreversible_given_critical_probability"])
            else:
                irreversible=bool(rng.random()<E["irreversible_noncritical_probability"])

            core_conflict=noisy_bool(rng,critical,E["normative_signal_accuracy"])
            world_risk=noisy_bool(rng,critical,E["world_signal_accuracy"])
            pressure_risk=noisy_bool(rng,critical,E["pressure_signal_accuracy"])
            uncertainty_high=noisy_bool(rng,critical or irreversible,E["uncertainty_signal_accuracy"])
            irreversible_cue=noisy_bool(rng,irreversible,0.78)

            if model=="severity_weighted_hierarchical":
                cfg=D[model]
                score=(
                    cfg["irreversibility_weight"]*int(irreversible_cue)
                    +cfg["core_conflict_weight"]*int(core_conflict)
                    +cfg["world_risk_weight"]*int(world_risk)
                    +cfg["uncertainty_weight"]*int(uncertainty_high)
                )
                threshold = th_i if irreversible_cue else th_r
            else:
                score=(
                    0.38*int(core_conflict)
                    +0.28*int(world_risk)
                    +0.16*int(pressure_risk)
                    +0.10*int(uncertainty_high)
                    +0.08*int(irreversible_cue)
                )
                if model=="hierarchical_recall_constrained":
                    threshold = th_i if irreversible_cue else th_r
                else:
                    threshold = th

            trigger = score>=threshold

            if trigger:
                reviews+=1
                if critical and irreversible:
                    tp_i+=1
                elif critical and not irreversible:
                    tp_r+=1
                elif not critical:
                    fp+=1
            else:
                if critical and irreversible:
                    fn_i+=1
                elif critical and not irreversible:
                    fn_r+=1

            protected=False
            if trigger:
                protected=bool(rng.random()<E["review_success_probability"])
                traj-=E["review_cost"]
                if not critical:
                    traj-=E["false_alarm_cost"]

            if critical and irreversible and not protected:
                irreversible_viol+=1
                traj-=E["missed_irreversible_penalty"]
            elif critical and not irreversible and not protected:
                reversible_miss+=1
                traj-=E["missed_reversible_penalty"]
            else:
                traj+=1.0

            # post-action updates
            if model=="global_recall_constrained":
                if trigger and critical: 
                    if irreversible: rtp_i+=1
                    else: rtp_r+=1
                elif (not trigger) and critical:
                    if irreversible: rfn_i+=1
                    else: rfn_r+=1

                if t>=D[model]["burn_in"]:
                    recall_all=(rtp_i+rtp_r)/(rtp_i+rtp_r+rfn_i+rfn_r)
                    if recall_all < D[model]["recall_floor"]:
                        th-=D[model]["adapt_rate"]
                    else:
                        th+=0.25*D[model]["adapt_rate"]
                    th=clip(th,D[model]["min_threshold"],D[model]["max_threshold"])

            elif model=="hierarchical_recall_constrained":
                if trigger and critical:
                    if irreversible: rtp_i+=1
                    else: rtp_r+=1
                elif (not trigger) and critical:
                    if irreversible: rfn_i+=1
                    else: rfn_r+=1

                if t>=D[model]["burn_in"]:
                    rec_i=rtp_i/(rtp_i+rfn_i)
                    rec_r=rtp_r/(rtp_r+rfn_r)

                    if rec_i < D[model]["irreversible_recall_floor"]:
                        th_i-=D[model]["adapt_rate"]
                    else:
                        th_i+=0.20*D[model]["adapt_rate"]

                    if rec_r < D[model]["reversible_recall_floor"]:
                        th_r-=D[model]["adapt_rate"]
                    else:
                        th_r+=0.20*D[model]["adapt_rate"]

                    th_i=clip(th_i,D[model]["min_threshold"],D[model]["max_threshold"])
                    th_r=clip(th_r,D[model]["min_threshold"],D[model]["max_threshold"])

        recall_i=tp_i/max(1,tp_i+fn_i)
        recall_r=tp_r/max(1,tp_r+fn_r)
        precision=(tp_i+tp_r)/max(1,tp_i+tp_r+fp)
        false_alarm=fp/SPEC["episodes"]
        trigger_rate=reviews/SPEC["episodes"]
        viol=irreversible_viol/max(1,tp_i+fn_i)
        rev_miss=reversible_miss/max(1,tp_r+fn_r)

        net_value=(
            0.40*recall_i
            +0.18*recall_r
            +0.16*precision
            +0.12*(1-false_alarm)
            +0.14*(1-viol)
        )

        rows.append({
            "seed":seed,
            "model":model,
            "irreversible_recall":recall_i,
            "reversible_recall":recall_r,
            "critical_precision":precision,
            "false_alarm_rate":false_alarm,
            "review_trigger_rate":trigger_rate,
            "irreversible_core_violation_rate":viol,
            "reversible_miss_rate":rev_miss,
            "net_vigilance_value":net_value,
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "final_threshold_irreversible": th_i if model in ("hierarchical_recall_constrained","severity_weighted_hierarchical") else th,
            "final_threshold_reversible": th_r if model in ("hierarchical_recall_constrained","severity_weighted_hierarchical") else th
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v270.csv",index=False)

summary=df.groupby("model").agg(
    irreversible_recall=("irreversible_recall","mean"),
    reversible_recall=("reversible_recall","mean"),
    critical_precision=("critical_precision","mean"),
    false_alarm_rate=("false_alarm_rate","mean"),
    review_trigger_rate=("review_trigger_rate","mean"),
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    reversible_miss_rate=("reversible_miss_rate","mean"),
    net_vigilance_value=("net_vigilance_value","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    final_threshold_irreversible=("final_threshold_irreversible","mean"),
    final_threshold_reversible=("final_threshold_reversible","mean")
).reset_index()
summary.to_csv(BASE/"summary_v270.csv",index=False)

S=summary.set_index("model")
f="fixed_threshold"
g="global_recall_constrained"
h="hierarchical_recall_constrained"
s="severity_weighted_hierarchical"

safe=S[S["irreversible_recall"]>=0.90]
safe_best=safe["net_vigilance_value"].idxmax() if len(safe) else None
best_hier=max(S.loc[h,"long_term_trajectory_value"],S.loc[s,"long_term_trajectory_value"])

checks={
    "hierarchical_meets_irreversible_floor": bool(S.loc[h,"irreversible_recall"]>=0.90),
    "hierarchical_meets_reversible_floor": bool(S.loc[h,"reversible_recall"]>=0.60),
    "hierarchical_lower_false_alarms_than_global": bool(S.loc[h,"false_alarm_rate"]<=S.loc[g,"false_alarm_rate"]-0.05),
    "hierarchical_no_more_irreversible_violations_than_global": bool(S.loc[h,"irreversible_core_violation_rate"]<=S.loc[g,"irreversible_core_violation_rate"]+0.01),
    "severity_hierarchical_meets_irreversible_floor": bool(S.loc[s,"irreversible_recall"]>=0.90),
    "best_safe_detector_net_value_is_hierarchical": bool(safe_best in (h,s)),
    "hierarchical_best_or_tied_trajectory": bool(best_hier>=S.loc[f,"long_term_trajectory_value"]-0.01)
}

(BASE/"acceptance_v270.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nSAFE DETECTORS:", list(safe.index))
print("BEST SAFE:", safe_best)
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
