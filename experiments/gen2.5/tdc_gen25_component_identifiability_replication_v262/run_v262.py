
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v262.json").read_text())
P=SPEC["parameters"]
rows=[]

def noisy_bool(rng, truth, acc):
    return truth if rng.random()<acc else (not truth)

models=list(SPEC["models"].keys())
n=SPEC["episodes_per_task"]

for seed in SPEC["seeds"]:
    for mi,model in enumerate(models):
        rng=np.random.default_rng(seed*100+mi)
        # 1. ethical incoherence detection
        ok_eth=0
        for _ in range(n):
            truth=bool(rng.random()<0.5)
            if model=="ablate_ethicalsense":
                acc=P["generic_salience_accuracy"]
            else:
                acc=P["ethical_detection_accuracy"]
                if model!="full_loop":
                    acc=max(0.5, acc-P["cross_component_compensation"]*0.25)
            pred=noisy_bool(rng,truth,acc)
            ok_eth+=int(pred==truth)

        # 2. transformation necessity
        ok_gap=0
        for _ in range(n):
            truth=bool(rng.random()<0.5)
            if model=="ablate_idealgap":
                acc=P["generic_change_signal_accuracy"]
            else:
                acc=P["idealgap_accuracy"]
                if model!="full_loop":
                    acc=max(0.5, acc-P["cross_component_compensation"]*0.20)
            pred=noisy_bool(rng,truth,acc)
            ok_gap+=int(pred==truth)

        # 3. false transformation consolidation
        false_cons=0
        for _ in range(n):
            cand_correct=bool(rng.random()<P["candidate_correct_probability"])
            if model=="ablate_shadow_validation":
                accepted=True if rng.random()<0.92 else False
            else:
                sv_acc=P["shadow_validation_accuracy"]
                shadow=noisy_bool(rng,cand_correct,sv_acc)
                accepted=shadow
            if accepted and not cand_correct:
                false_cons+=1

        # 4. anti-core reward capture
        capture=0
        for _ in range(n):
            anti_core=bool(rng.random()<P["anti_core_reward_probability"])
            if anti_core:
                if model=="ablate_core_constraint":
                    follow_reward=bool(rng.random()<P["ablated_core_reward_follow"])
                else:
                    # full and non-target ablations retain partial core resistance
                    resist=P["full_core_resistance"]
                    if model!="full_loop":
                        resist=max(0.5,resist-P["cross_component_compensation"]*0.30)
                    follow_reward=bool(rng.random()<(1-resist))
                capture += int(follow_reward)

        eth_acc=ok_eth/n
        gap_acc=ok_gap/n
        false_rate=false_cons/n
        cap_rate=capture/n
        overall=0.30*eth_acc+0.25*gap_acc+0.25*(1-false_rate)+0.20*(1-cap_rate)

        rows.append({
            "seed":seed,"model":model,
            "ethical_detection_accuracy":eth_acc,
            "transformation_decision_accuracy":gap_acc,
            "false_consolidation_rate":false_rate,
            "anti_core_capture_rate":cap_rate,
            "overall_component_score":overall
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v262.csv",index=False)

summary=df.groupby("model").agg(
    ethical_detection_accuracy=("ethical_detection_accuracy","mean"),
    transformation_decision_accuracy=("transformation_decision_accuracy","mean"),
    false_consolidation_rate=("false_consolidation_rate","mean"),
    anti_core_capture_rate=("anti_core_capture_rate","mean"),
    overall_component_score=("overall_component_score","mean")
).reset_index()
summary.to_csv(BASE/"summary_v262.csv",index=False)

S=summary.set_index("model")
f="full_loop"; ae="ablate_ethicalsense"; ai="ablate_idealgap"; ash="ablate_shadow_validation"; ac="ablate_core_constraint"

task_cols={
    "ethical":("ethical_detection_accuracy",ae,True),
    "ideal":("transformation_decision_accuracy",ai,True),
    "shadow":("false_consolidation_rate",ash,False),
    "core":("anti_core_capture_rate",ac,False)
}

non_target_ok=True
specificity_ok=True
for _,(col,target,higher_better) in task_cols.items():
    full=S.loc[f,col]
    non_targets=[m for m in [ae,ai,ash,ac] if m!=target]
    within=sum(abs(S.loc[m,col]-full)<=0.10 for m in non_targets)
    if within<2:
        non_target_ok=False
    vals={m:S.loc[m,col] for m in [ae,ai,ash,ac]}
    ordered=sorted(vals.items(), key=lambda kv: kv[1], reverse=not higher_better)
    worst_two=[m for m,_ in ordered[:2]]
    if target not in worst_two:
        specificity_ok=False

checks={
    "ethicalsense_specific_failure": bool(S.loc[ae,"ethical_detection_accuracy"]<=S.loc[f,"ethical_detection_accuracy"]-0.10),
    "idealgap_specific_failure": bool(S.loc[ai,"transformation_decision_accuracy"]<=S.loc[f,"transformation_decision_accuracy"]-0.10),
    "shadow_specific_failure": bool(S.loc[ash,"false_consolidation_rate"]>=S.loc[f,"false_consolidation_rate"]+0.10),
    "core_specific_failure": bool(S.loc[ac,"anti_core_capture_rate"]>=S.loc[f,"anti_core_capture_rate"]+0.15),
    "full_best_overall": bool(all(S.loc[f,"overall_component_score"]>S.loc[m,"overall_component_score"] for m in [ae,ai,ash,ac])),
    "non_target_ablations_limited": bool(non_target_ok),
    "task_specificity": bool(specificity_ok)
}

(BASE/"acceptance_v262.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
