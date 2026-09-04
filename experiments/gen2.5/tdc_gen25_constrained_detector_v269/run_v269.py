
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v269.json").read_text())
E=SPEC["environment"]
D=SPEC["detectors"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)

        tp=fp=fn=tn=0
        critical_n=0
        irreversible_viol=0
        reviews=0
        traj=0.0

        if model=="fixed_threshold":
            threshold=D["fixed_threshold"]["threshold"]
        elif model=="severity_weighted_detector":
            threshold=D["severity_weighted_detector"]["threshold"]
        else:
            threshold=D[model]["initial_threshold"]

        running_tp=2
        running_fp=2
        running_fn=2

        for t in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_action_probability"])

            core_conflict=noisy_bool(rng,critical,E["normative_signal_accuracy"])
            world_risk=noisy_bool(rng,critical,E["world_signal_accuracy"])
            pressure_risk=noisy_bool(rng,critical,E["pressure_signal_accuracy"])
            uncertainty_high=noisy_bool(rng,critical or irreversible,E["uncertainty_signal_accuracy"])

            if model=="severity_weighted_detector":
                cfg=D["severity_weighted_detector"]
                score=(
                    cfg["irreversibility_weight"]*int(irreversible)
                    +cfg["core_conflict_weight"]*int(core_conflict)
                    +cfg["world_risk_weight"]*int(world_risk)
                    +cfg["uncertainty_weight"]*int(uncertainty_high)
                )
            else:
                score=(
                    0.40*int(core_conflict)
                    +0.30*int(world_risk)
                    +0.18*int(pressure_risk)
                    +0.12*int(uncertainty_high)
                )

            trigger=score>=threshold

            if trigger:
                reviews+=1
                if critical: tp+=1
                else: fp+=1
            else:
                if critical: fn+=1
                else: tn+=1

            if critical:
                critical_n+=1

            protected=False
            if trigger:
                protected=bool(rng.random()<E["review_success_probability"])
                traj-=E["review_cost"]
                if not critical:
                    traj-=E["false_alarm_cost"]

            if critical and irreversible and not protected:
                irreversible_viol+=1
                traj-=E["missed_critical_penalty"]
            else:
                traj+=1.0

            # post-action adaptive updates only
            if model in ("precision_first_adaptive","recall_constrained_adaptive"):
                if trigger and critical: running_tp+=1
                elif trigger and not critical: running_fp+=1
                elif (not trigger) and critical: running_fn+=1

                precision=running_tp/(running_tp+running_fp)
                recall=running_tp/(running_tp+running_fn)

                if model=="precision_first_adaptive":
                    cfg=D[model]
                    if precision<cfg["target_precision"]:
                        threshold+=cfg["adapt_rate"]
                    elif recall<cfg["target_recall"]:
                        threshold-=cfg["adapt_rate"]
                    threshold=clip(threshold,cfg["min_threshold"],cfg["max_threshold"])

                else:
                    cfg=D[model]
                    if t>=cfg["burn_in"]:
                        # hard safety constraint has priority
                        if recall < cfg["recall_floor"]:
                            threshold -= cfg["adapt_rate"]
                        elif precision < cfg["precision_target"] and recall > cfg["recall_floor"]+0.04:
                            threshold += cfg["adapt_rate"]
                    threshold=clip(threshold,cfg["min_threshold"],cfg["max_threshold"])

        recall=tp/max(1,tp+fn)
        precision=tp/max(1,tp+fp)
        false_alarm=fp/SPEC["episodes"]
        trigger_rate=reviews/SPEC["episodes"]
        viol=irreversible_viol/max(1,critical_n)
        f1=2*precision*recall/max(1e-9,precision+recall)

        net_value=(
            0.42*recall
            +0.24*precision
            +0.18*(1-false_alarm)
            +0.16*(1-viol)
        )

        rows.append({
            "seed":seed,
            "model":model,
            "critical_recall":recall,
            "critical_precision":precision,
            "f1_like":f1,
            "false_alarm_rate":false_alarm,
            "review_trigger_rate":trigger_rate,
            "irreversible_core_violation_rate":viol,
            "net_vigilance_value":net_value,
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "final_threshold":threshold
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v269.csv",index=False)

summary=df.groupby("model").agg(
    critical_recall=("critical_recall","mean"),
    critical_precision=("critical_precision","mean"),
    f1_like=("f1_like","mean"),
    false_alarm_rate=("false_alarm_rate","mean"),
    review_trigger_rate=("review_trigger_rate","mean"),
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    net_vigilance_value=("net_vigilance_value","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    final_threshold=("final_threshold","mean")
).reset_index()
summary.to_csv(BASE/"summary_v269.csv",index=False)

S=summary.set_index("model")
f="fixed_threshold"
p="precision_first_adaptive"
r="recall_constrained_adaptive"
s="severity_weighted_detector"

safe = S[S["critical_recall"]>=0.80]
safe_best = safe["net_vigilance_value"].idxmax() if len(safe) else None

checks={
    "recall_constrained_meets_floor": bool(S.loc[r,"critical_recall"]>=0.80),
    "recall_constrained_beats_precision_first_recall": bool(S.loc[r,"critical_recall"]>=S.loc[p,"critical_recall"]+0.30),
    "recall_constrained_lower_false_alarms_than_fixed": bool(S.loc[r,"false_alarm_rate"]<=S.loc[f,"false_alarm_rate"]-0.03),
    "recall_constrained_not_more_violations_than_fixed": bool(S.loc[r,"irreversible_core_violation_rate"]<=S.loc[f,"irreversible_core_violation_rate"]+0.01),
    "severity_weighted_meets_floor": bool(S.loc[s,"critical_recall"]>=0.80),
    "best_safe_detector_net_value": bool(safe_best in (r,s)),
    "safety_constraint_prevents_threshold_collapse": bool(S.loc[r,"final_threshold"]<=0.65)
}
(BASE/"acceptance_v269.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nSAFE DETECTORS:", list(safe.index))
print("BEST SAFE:", safe_best)
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
