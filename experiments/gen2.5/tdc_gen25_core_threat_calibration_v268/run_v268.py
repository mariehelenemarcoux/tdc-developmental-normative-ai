
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v268.json").read_text())
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

        threshold = D["adaptive_detector"]["initial_threshold"] if model=="adaptive_detector" else D.get(model,{}).get("threshold",0.58)

        running_tp=1
        running_fp=1
        running_fn=1

        for _ in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_action_probability"])

            core_conflict = noisy_bool(rng, critical, E["normative_signal_accuracy"])
            world_risk = noisy_bool(rng, critical, E["world_signal_accuracy"])
            pressure_risk = noisy_bool(rng, critical, E["pressure_signal_accuracy"])
            uncertainty_high = noisy_bool(rng, critical or irreversible, E["uncertainty_signal_accuracy"])

            if model=="fixed_threshold":
                score=(
                    0.40*int(core_conflict)
                    +0.30*int(world_risk)
                    +0.18*int(pressure_risk)
                    +0.12*int(uncertainty_high)
                )

            elif model=="calibrated_score":
                # reliability-weighted score with irreversibility as explicit modifier
                score=(
                    0.36*E["normative_signal_accuracy"]*int(core_conflict)
                    +0.28*E["world_signal_accuracy"]*int(world_risk)
                    +0.16*E["pressure_signal_accuracy"]*int(pressure_risk)
                    +0.12*E["uncertainty_signal_accuracy"]*int(uncertainty_high)
                    +0.18*int(irreversible)
                )
                score=clip(score,0,1)

            elif model=="structural_score":
                c=(0.60*int(core_conflict)+0.40*int(world_risk))
                u=(0.55*int(uncertainty_high)+0.45*(1-E["world_signal_accuracy"]))
                i=(1.0 if irreversible else 0.35)
                score=i*c*u

            else:
                # adaptive calibrated score
                base=(
                    0.36*E["normative_signal_accuracy"]*int(core_conflict)
                    +0.28*E["world_signal_accuracy"]*int(world_risk)
                    +0.16*E["pressure_signal_accuracy"]*int(pressure_risk)
                    +0.12*E["uncertainty_signal_accuracy"]*int(uncertainty_high)
                    +0.18*int(irreversible)
                )
                score=clip(base,0,1)

            trigger = score>=threshold

            if trigger:
                reviews+=1
                if critical: tp+=1
                else: fp+=1
            else:
                if critical: fn+=1
                else: tn+=1

            if critical:
                critical_n+=1

            # review outcome
            protected=False
            if trigger:
                if rng.random()<E["review_success_probability"]:
                    protected=True
                    traj += 0.20
                traj -= E["review_step_cost"]*3.5
                if not critical:
                    traj -= E["false_alarm_cost"]

            if critical and irreversible and not protected:
                irreversible_viol+=1
                traj -= E["missed_critical_penalty"]
            else:
                traj += 1.0

            # post-action adaptive threshold update
            if model=="adaptive_detector":
                if trigger and critical: running_tp+=1
                elif trigger and not critical: running_fp+=1
                elif (not trigger) and critical: running_fn+=1

                precision=running_tp/(running_tp+running_fp)
                recall=running_tp/(running_tp+running_fn)

                if precision < D["adaptive_detector"]["target_precision"]:
                    threshold += D["adaptive_detector"]["adapt_rate"]
                elif recall < D["adaptive_detector"]["target_recall"]:
                    threshold -= D["adaptive_detector"]["adapt_rate"]

                threshold=clip(
                    threshold,
                    D["adaptive_detector"]["min_threshold"],
                    D["adaptive_detector"]["max_threshold"]
                )

        recall=tp/max(1,tp+fn)
        precision=tp/max(1,tp+fp)
        false_alarm=fp/SPEC["episodes"]
        miss_rate=fn/max(1,tp+fn)
        trigger_rate=reviews/SPEC["episodes"]
        viol=irreversible_viol/max(1,critical_n)
        f1=2*precision*recall/max(1e-9,precision+recall)

        net_value=(
            0.38*recall
            +0.28*precision
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
            "miss_rate":miss_rate,
            "review_trigger_rate":trigger_rate,
            "irreversible_core_violation_rate":viol,
            "net_vigilance_value":net_value,
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "final_threshold":threshold
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v268.csv",index=False)

summary=df.groupby("model").agg(
    critical_recall=("critical_recall","mean"),
    critical_precision=("critical_precision","mean"),
    f1_like=("f1_like","mean"),
    false_alarm_rate=("false_alarm_rate","mean"),
    miss_rate=("miss_rate","mean"),
    review_trigger_rate=("review_trigger_rate","mean"),
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    net_vigilance_value=("net_vigilance_value","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    final_threshold=("final_threshold","mean")
).reset_index()
summary.to_csv(BASE/"summary_v268.csv",index=False)

S=summary.set_index("model")
f="fixed_threshold"; c="calibrated_score"; s="structural_score"; a="adaptive_detector"

checks={
    "calibrated_beats_fixed_precision": bool(S.loc[c,"critical_precision"]>=S.loc[f,"critical_precision"]+0.08),
    "calibrated_keeps_recall": bool(S.loc[c,"critical_recall"]>=S.loc[f,"critical_recall"]-0.03),
    "structural_reduces_false_alarms": bool(S.loc[s,"false_alarm_rate"]<=S.loc[f,"false_alarm_rate"]-0.08),
    "adaptive_best_f1_like": bool(S.loc[a,"f1_like"]>=S["f1_like"].max()-1e-12),
    "adaptive_reduces_core_violations": bool(S.loc[a,"irreversible_core_violation_rate"]<=S.loc[f,"irreversible_core_violation_rate"]-0.03),
    "adaptive_improves_net_value": bool(S.loc[a,"net_vigilance_value"]>=S.loc[f,"net_vigilance_value"]+0.08),
    "adaptive_false_alarm_below_0_15": bool(S.loc[a,"false_alarm_rate"]<=0.15)
}
(BASE/"acceptance_v268.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
