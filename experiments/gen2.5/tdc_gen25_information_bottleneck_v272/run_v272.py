
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v272.json").read_text())
D=SPEC["detector"]
E=SPEC["environment"]
rows=[]

def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)
def clip(x,a,b): return max(a,min(b,x))

for q in SPEC["quality_levels"]:
    for seed in SPEC["seeds"]:
        rng=np.random.default_rng(seed*1000+int(q*100))

        tp_i=fn_i=tp_r=fn_r=fp=0
        probes=0
        reviews=0
        ir_viol=0
        rev_miss=0
        traj=0.0

        # map quality to signal channels; keep pressure fixed
        norm_acc=q
        world_acc=clip(q-0.02,0.50,0.99)
        uncertainty_acc=clip(q+0.02,0.50,0.99)
        irreversibility_acc=clip(q+0.05,0.50,0.99)
        probe_acc=clip(q+D["probe_signal_accuracy_bonus"],0.55,0.98)

        for _ in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=(
                bool(rng.random()<E["irreversible_given_critical_probability"])
                if critical else
                bool(rng.random()<E["irreversible_noncritical_probability"])
            )

            core_conflict=noisy_bool(rng,critical,norm_acc)
            world_risk=noisy_bool(rng,critical,world_acc)
            pressure_risk=noisy_bool(rng,critical,E["pressure_signal_accuracy"])
            uncertainty_high=noisy_bool(rng,critical or irreversible,uncertainty_acc)
            irreversible_cue=noisy_bool(rng,irreversible,irreversibility_acc)

            score=(
                0.38*int(core_conflict)
                +0.28*int(world_risk)
                +0.16*int(pressure_risk)
                +0.10*int(uncertainty_high)
                +0.08*int(irreversible_cue)
            )
            threshold=D["threshold_irreversible"] if irreversible_cue else D["threshold_reversible"]

            boundary_uncertain=abs(score-threshold)<=D["boundary_margin"]
            if irreversible_cue and boundary_uncertain:
                probes+=1
                probe_positive=noisy_bool(rng,critical,probe_acc)
                score=clip(score+(0.22 if probe_positive else -0.18),0,1)
                traj-=D["probe_cost"]

            trigger=score>=threshold

            if trigger:
                reviews+=1
                if critical and irreversible: tp_i+=1
                elif critical and not irreversible: tp_r+=1
                elif not critical: fp+=1
            else:
                if critical and irreversible: fn_i+=1
                elif critical and not irreversible: fn_r+=1

            protected=False
            if trigger:
                protected=bool(rng.random()<D["review_success_probability"])
                traj-=E["review_cost"]
                if not critical:
                    traj-=E["false_alarm_cost"]

            if critical and irreversible and not protected:
                ir_viol+=1
                traj-=E["missed_irreversible_penalty"]
            elif critical and not irreversible and not protected:
                rev_miss+=1
                traj-=E["missed_reversible_penalty"]
            else:
                traj+=1.0

        recall_i=tp_i/max(1,tp_i+fn_i)
        recall_r=tp_r/max(1,tp_r+fn_r)
        precision=(tp_i+tp_r)/max(1,tp_i+tp_r+fp)
        false_alarm=fp/SPEC["episodes"]
        probe_rate=probes/SPEC["episodes"]
        viol=ir_viol/max(1,tp_i+fn_i)

        net=(
            0.42*recall_i
            +0.16*recall_r
            +0.14*precision
            +0.10*(1-false_alarm)
            +0.12*(1-viol)
            +0.06*(1-probe_rate)
        )

        rows.append({
            "quality":q,
            "seed":seed,
            "irreversible_recall":recall_i,
            "reversible_recall":recall_r,
            "critical_precision":precision,
            "false_alarm_rate":false_alarm,
            "probe_rate":probe_rate,
            "irreversible_core_violation_rate":viol,
            "net_vigilance_value":net,
            "long_term_trajectory_value":traj/SPEC["episodes"]
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v272.csv",index=False)

summary=df.groupby("quality").agg(
    irreversible_recall=("irreversible_recall","mean"),
    reversible_recall=("reversible_recall","mean"),
    critical_precision=("critical_precision","mean"),
    false_alarm_rate=("false_alarm_rate","mean"),
    probe_rate=("probe_rate","mean"),
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    net_vigilance_value=("net_vigilance_value","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean")
).reset_index()
summary.to_csv(BASE/"summary_v272.csv",index=False)

# monotonic trend count
vals=summary["irreversible_recall"].tolist()
monotonic_steps=sum(vals[i+1]>=vals[i] for i in range(len(vals)-1))
recall_monotonic_overall = monotonic_steps >= len(vals)-2

safe=summary[summary["irreversible_recall"]>=0.90]
first_safe=safe.iloc[0] if len(safe) else None

def getq(q,col):
    return float(summary.loc[np.isclose(summary["quality"],q),col].iloc[0])

checks={
    "recall_monotonic_overall": bool(recall_monotonic_overall),
    "quality_0_85_reaches_0_90_recall": bool(getq(0.85,"irreversible_recall")>=0.90),
    "quality_0_90_reaches_0_90_recall": bool(getq(0.90,"irreversible_recall")>=0.90),
    "quality_threshold_exists": bool(len(safe)>0),
    "false_alarm_below_0_20_at_first_safe_level": bool(first_safe is not None and first_safe["false_alarm_rate"]<=0.20),
    "violations_drop_with_quality": bool(getq(0.90,"irreversible_core_violation_rate")<=getq(0.70,"irreversible_core_violation_rate")-0.05),
    "net_value_not_collapsed_at_safe_level": bool(first_safe is not None and first_safe["net_vigilance_value"]>=0.70)
}
(BASE/"acceptance_v272.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nFIRST SAFE LEVEL:", None if first_safe is None else float(first_safe["quality"]))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
