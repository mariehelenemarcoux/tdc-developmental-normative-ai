
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v267.json").read_text())
E=SPEC["environment"]
R=SPEC["review"]
TH=SPEC["thresholds"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)

        critical_n=0
        irreversible_viol=0
        review_triggers=0
        false_alarms=0
        unnecessary_abstentions=0
        review_steps=0
        traj=0.0
        correct_n=0
        total_n=0

        for _ in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_action_probability"])
            pressure=bool(rng.random()<E["external_pressure_probability"])
            safe_reversible=bool(rng.random()<E["safe_reversible_option_probability"])

            should_block=critical

            core_signal=noisy_bool(rng,should_block,E["normative_signal_accuracy"])
            world_signal=noisy_bool(rng,should_block,E["world_signal_accuracy"])
            pressure_pushes_risk=bool(pressure and rng.random()<E["pressure_wrong_probability"])

            risk_score=clip(
                0.46*int(core_signal)
                +0.34*int(world_signal)
                +0.20*int(pressure_pushes_risk),
                0,1
            )

            action_block = risk_score >= 0.50
            abstain=False

            if model!="no_emergency":
                threshold=TH[model]
                do_review=risk_score>=threshold
                if do_review:
                    review_triggers+=1
                    if not critical:
                        false_alarms+=1
                        traj-=E["false_alarm_opportunity_cost"]

                    votes=[]
                    found_reversible=False
                    for c in range(R["max_cycles"]):
                        review_steps+=1
                        check_acc=clip(0.69+0.02*c,0.55,0.87)
                        vote=noisy_bool(rng,should_block,check_acc)
                        votes.append(1 if vote else 0)

                        if safe_reversible and rng.random()<(0.20+0.05*c):
                            found_reversible=True

                        consensus=sum(votes)/len(votes)
                        if len(votes)>=3 and (consensus>=R["consensus_threshold"] or consensus<=1-R["consensus_threshold"]):
                            break
                        if found_reversible and len(votes)>=2:
                            break

                    if not critical:
                        traj-=E["delay_cost_noncritical"]*len(votes)/R["max_cycles"]

                    if found_reversible:
                        action_block = True if critical else False
                    else:
                        consensus=sum(votes)/len(votes)
                        if consensus>=0.5:
                            if irreversible and abs(consensus-0.5)<0.18:
                                abstain=True
                                if not critical:
                                    unnecessary_abstentions+=1
                                    traj-=E["unnecessary_abstention_cost"]
                            else:
                                action_block=True
                        else:
                            action_block=False

            total_n+=1
            if critical:
                critical_n+=1

            if abstain:
                traj-=0.12
            else:
                correct=(action_block==should_block)
                correct_n+=int(correct)

                if critical and irreversible and not action_block:
                    irreversible_viol+=1
                    traj-=E["severe_core_violation_penalty"]
                elif correct:
                    traj+=1.0
                else:
                    traj-=E["normal_error_penalty"]

        traj -= review_steps*E["review_step_cost"]

        viol=irreversible_viol/max(1,critical_n)
        trigger_rate=review_triggers/SPEC["episodes"]
        false_alarm_rate=false_alarms/SPEC["episodes"]
        unnecessary_abst=unnecessary_abstentions/SPEC["episodes"]
        steps_per_ep=review_steps/SPEC["episodes"]
        acc=correct_n/max(1,total_n-unnecessary_abstentions)

        # Explicit net vigilance score: benefit from avoided irreversible violations,
        # minus review burden, false alarms, and abstention.
        net_value = (
            1.0
            - 2.2*viol
            - 1.4*false_alarm_rate
            - 1.0*unnecessary_abst
            - 0.12*steps_per_ep
        )

        integrated = (
            0.30*acc
            +0.25*(1-viol)
            +0.20*(1-false_alarm_rate)
            +0.10*(1-unnecessary_abst)
            +0.15*clip((traj/SPEC["episodes"]+1)/2,0,1)
        )

        rows.append({
            "seed":seed,
            "model":model,
            "irreversible_core_violation_rate":viol,
            "false_alarm_rate":false_alarm_rate,
            "review_trigger_rate":trigger_rate,
            "unnecessary_abstention_rate":unnecessary_abst,
            "review_steps_per_episode":steps_per_ep,
            "decision_accuracy":acc,
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "net_vigilance_value":net_value,
            "integrated_score":integrated
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v267.csv",index=False)

summary=df.groupby("model").agg(
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    false_alarm_rate=("false_alarm_rate","mean"),
    review_trigger_rate=("review_trigger_rate","mean"),
    unnecessary_abstention_rate=("unnecessary_abstention_rate","mean"),
    review_steps_per_episode=("review_steps_per_episode","mean"),
    decision_accuracy=("decision_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    net_vigilance_value=("net_vigilance_value","mean"),
    integrated_score=("integrated_score","mean")
).reset_index()
summary.to_csv(BASE/"summary_v267.csv",index=False)

S=summary.set_index("model")
n="no_emergency"; c="conservative_review"; b="balanced_review"; h="hypersensitive_review"

checks={
    "balanced_reduces_violations": bool(S.loc[b,"irreversible_core_violation_rate"]<=S.loc[n,"irreversible_core_violation_rate"]-0.05),
    "balanced_false_alarm_below_hyper": bool(S.loc[b,"false_alarm_rate"]<=S.loc[h,"false_alarm_rate"]-0.10),
    "hyper_pays_vigilance_cost": bool(S.loc[h,"net_vigilance_value"]<=S.loc[b,"net_vigilance_value"]-0.03),
    "conservative_underprotects": bool(S.loc[c,"irreversible_core_violation_rate"]>=S.loc[b,"irreversible_core_violation_rate"]+0.02),
    "balanced_best_net_value": bool(S.loc[b,"net_vigilance_value"]>=S["net_vigilance_value"].max()-1e-12),
    "balanced_no_chronic_abstention": bool(S.loc[b,"unnecessary_abstention_rate"]<=0.08),
    "u_shaped_tradeoff": bool(
        S.loc[c,"irreversible_core_violation_rate"]>S.loc[b,"irreversible_core_violation_rate"]
        and S.loc[h,"false_alarm_rate"]>S.loc[b,"false_alarm_rate"]
    )
}
(BASE/"acceptance_v267.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
