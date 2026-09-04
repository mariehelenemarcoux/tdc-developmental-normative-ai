
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v271.json").read_text())
E=SPEC["environment"]
D=SPEC["detector"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)

        th_i=D["initial_threshold_irreversible"]
        th_r=D["initial_threshold_reversible"]

        tp_i=fn_i=0
        tp_r=fn_r=0
        fp=0
        reviews=0
        probes=0
        irreversible_viol=0
        reversible_miss=0
        traj=0.0

        rtp_i=rfn_i=2
        rtp_r=rfn_r=2

        for t in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible = (
                bool(rng.random()<E["irreversible_given_critical_probability"])
                if critical else
                bool(rng.random()<E["irreversible_noncritical_probability"])
            )

            core_conflict=noisy_bool(rng,critical,E["normative_signal_accuracy"])
            world_risk=noisy_bool(rng,critical,E["world_signal_accuracy"])
            pressure_risk=noisy_bool(rng,critical,E["pressure_signal_accuracy"])
            uncertainty_high=noisy_bool(rng,critical or irreversible,E["uncertainty_signal_accuracy"])
            irreversible_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_accuracy"])

            score=(
                0.38*int(core_conflict)
                +0.28*int(world_risk)
                +0.16*int(pressure_risk)
                +0.10*int(uncertainty_high)
                +0.08*int(irreversible_cue)
            )

            threshold=th_i if irreversible_cue else th_r

            # Decide whether to probe.
            do_probe=False
            boundary_uncertain = abs(score-threshold) <= D["boundary_margin"]

            running_ir_recall=rtp_i/max(1,rtp_i+rfn_i)

            if model=="hierarchical_boundary_probe":
                do_probe = irreversible_cue and boundary_uncertain

            elif model=="hierarchical_recall_guard_probe":
                guard_active = running_ir_recall < (D["irreversible_recall_floor"] + D["recall_guard_margin"])
                do_probe = irreversible_cue and (boundary_uncertain or guard_active)

            elif model=="global_probe":
                do_probe = irreversible_cue and (rng.random()<0.78)

            if do_probe:
                probes+=1
                # Independent noisy evidence: whether this is a critical Core threat.
                probe_positive=noisy_bool(rng,critical,E["probe_signal_accuracy"])
                if probe_positive:
                    score=clip(score+0.22,0,1)
                else:
                    score=clip(score-0.18,0,1)
                traj-=E["probe_cost"]

            trigger=score>=threshold

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

            # post-action audit stats and threshold adaptation
            if trigger and critical:
                if irreversible:
                    rtp_i+=1
                else:
                    rtp_r+=1
            elif (not trigger) and critical:
                if irreversible:
                    rfn_i+=1
                else:
                    rfn_r+=1

            if t>=D["burn_in"]:
                rec_i=rtp_i/(rtp_i+rfn_i)
                rec_r=rtp_r/(rtp_r+rfn_r)

                if rec_i < D["irreversible_recall_floor"]:
                    th_i-=D["adapt_rate"]
                else:
                    th_i+=0.20*D["adapt_rate"]

                if rec_r < D["reversible_recall_floor"]:
                    th_r-=D["adapt_rate"]
                else:
                    th_r+=0.20*D["adapt_rate"]

                th_i=clip(th_i,D["min_threshold"],D["max_threshold"])
                th_r=clip(th_r,D["min_threshold"],D["max_threshold"])

        recall_i=tp_i/max(1,tp_i+fn_i)
        recall_r=tp_r/max(1,tp_r+fn_r)
        precision=(tp_i+tp_r)/max(1,tp_i+tp_r+fp)
        false_alarm=fp/SPEC["episodes"]
        probe_rate=probes/SPEC["episodes"]
        trigger_rate=reviews/SPEC["episodes"]
        viol=irreversible_viol/max(1,tp_i+fn_i)
        rev_miss=reversible_miss/max(1,tp_r+fn_r)

        net_value=(
            0.42*recall_i
            +0.16*recall_r
            +0.14*precision
            +0.10*(1-false_alarm)
            +0.12*(1-viol)
            +0.06*(1-probe_rate)
        )

        rows.append({
            "seed":seed,
            "model":model,
            "irreversible_recall":recall_i,
            "reversible_recall":recall_r,
            "critical_precision":precision,
            "false_alarm_rate":false_alarm,
            "probe_rate":probe_rate,
            "review_trigger_rate":trigger_rate,
            "irreversible_core_violation_rate":viol,
            "reversible_miss_rate":rev_miss,
            "net_vigilance_value":net_value,
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "final_threshold_irreversible":th_i,
            "final_threshold_reversible":th_r
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v271.csv",index=False)

summary=df.groupby("model").agg(
    irreversible_recall=("irreversible_recall","mean"),
    reversible_recall=("reversible_recall","mean"),
    critical_precision=("critical_precision","mean"),
    false_alarm_rate=("false_alarm_rate","mean"),
    probe_rate=("probe_rate","mean"),
    review_trigger_rate=("review_trigger_rate","mean"),
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    reversible_miss_rate=("reversible_miss_rate","mean"),
    net_vigilance_value=("net_vigilance_value","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    final_threshold_irreversible=("final_threshold_irreversible","mean"),
    final_threshold_reversible=("final_threshold_reversible","mean")
).reset_index()
summary.to_csv(BASE/"summary_v271.csv",index=False)

S=summary.set_index("model")
n="hierarchical_no_probe"
b="hierarchical_boundary_probe"
g="hierarchical_recall_guard_probe"
gp="global_probe"

safe=S[S["irreversible_recall"]>=0.90]
safe_best=safe["net_vigilance_value"].idxmax() if len(safe) else None

checks={
    "guard_probe_meets_irreversible_floor": bool(S.loc[g,"irreversible_recall"]>=0.90),
    "boundary_probe_improves_recall": bool(S.loc[b,"irreversible_recall"]>=S.loc[n,"irreversible_recall"]+0.03),
    "guard_probe_improves_recall": bool(S.loc[g,"irreversible_recall"]>=S.loc[n,"irreversible_recall"]+0.05),
    "guard_probe_false_alarm_not_explosive": bool(S.loc[g,"false_alarm_rate"]<=S.loc[n,"false_alarm_rate"]+0.05),
    "guard_probe_better_violations": bool(S.loc[g,"irreversible_core_violation_rate"]<=S.loc[n,"irreversible_core_violation_rate"]-0.04),
    "guard_probe_more_efficient_than_global": bool(S.loc[g,"probe_rate"]<=S.loc[gp,"probe_rate"]-0.10),
    "best_safe_net_value_is_targeted": bool(safe_best in (b,g))
}
(BASE/"acceptance_v271.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nSAFE DETECTORS:", list(safe.index))
print("BEST SAFE:", safe_best)
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
