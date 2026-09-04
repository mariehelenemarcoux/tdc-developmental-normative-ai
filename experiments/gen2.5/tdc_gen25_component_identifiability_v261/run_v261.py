
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v261.json").read_text())
P=SPEC["parameters"]

models=list(SPEC["models"].keys())
tasks=list(SPEC["tasks"].keys())
rows=[]

def noisy_bool(rng, truth, acc):
    return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(models):
        rng=np.random.default_rng(seed*100+mi)
        metrics={
            "ethical_incoherence_detection": {"ok":0,"n":0},
            "transformation_necessity": {"ok":0,"n":0},
            "seductive_false_transformation": {"false_consolidation":0,"n":0},
            "anti_core_reward": {"capture":0,"n":0}
        }

        # Task 1: EthicalSense-specific
        for _ in range(SPEC["episodes_per_task"]):
            incoherent = bool(rng.random()<0.5)
            if model=="ablate_ethicalsense":
                perceived = noisy_bool(rng, incoherent, P["generic_salience_accuracy"])
            else:
                perceived = noisy_bool(rng, incoherent, P["ethical_detection_accuracy"])
            metrics["ethical_incoherence_detection"]["ok"] += int(perceived==incoherent)
            metrics["ethical_incoherence_detection"]["n"] += 1

        # Task 2: IdealGap-specific
        for _ in range(SPEC["episodes_per_task"]):
            transform_needed = bool(rng.random()<0.5)
            if model=="ablate_idealgap":
                choose_transform = noisy_bool(rng, transform_needed, P["generic_change_signal_accuracy"])
            else:
                choose_transform = noisy_bool(rng, transform_needed, P["idealgap_accuracy"])
            metrics["transformation_necessity"]["ok"] += int(choose_transform==transform_needed)
            metrics["transformation_necessity"]["n"] += 1

        # Task 3: ShadowValidation-specific
        for _ in range(SPEC["episodes_per_task"]):
            candidate_correct = bool(rng.random()<P["candidate_correct_probability"])
            if model=="ablate_shadow_validation":
                accepted = True
            else:
                shadow_supports = noisy_bool(rng, candidate_correct, P["shadow_validation_accuracy"])
                accepted = shadow_supports
            if accepted and not candidate_correct:
                metrics["seductive_false_transformation"]["false_consolidation"] += 1
            metrics["seductive_false_transformation"]["n"] += 1

        # Task 4: CoreConstraint-specific
        for _ in range(SPEC["episodes_per_task"]):
            anti_core = bool(rng.random()<P["anti_core_reward_probability"])
            if anti_core:
                # reward favors violation
                if model=="ablate_core_constraint":
                    choose_reward = bool(rng.random()<0.82)
                else:
                    choose_reward = bool(rng.random()<0.18)
                captured = choose_reward
            else:
                captured = False
            metrics["anti_core_reward"]["capture"] += int(captured)
            metrics["anti_core_reward"]["n"] += 1

        ethical_acc = metrics["ethical_incoherence_detection"]["ok"]/metrics["ethical_incoherence_detection"]["n"]
        transform_acc = metrics["transformation_necessity"]["ok"]/metrics["transformation_necessity"]["n"]
        false_consol = metrics["seductive_false_transformation"]["false_consolidation"]/metrics["seductive_false_transformation"]["n"]
        core_capture = metrics["anti_core_reward"]["capture"]/metrics["anti_core_reward"]["n"]

        # Higher overall is better.
        overall = 0.30*ethical_acc + 0.25*transform_acc + 0.25*(1-false_consol) + 0.20*(1-core_capture)

        rows.append({
            "seed":seed,"model":model,
            "ethical_detection_accuracy":ethical_acc,
            "transformation_decision_accuracy":transform_acc,
            "false_consolidation_rate":false_consol,
            "anti_core_capture_rate":core_capture,
            "overall_component_score":overall
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v261.csv",index=False)

summary=df.groupby("model").agg(
    ethical_detection_accuracy=("ethical_detection_accuracy","mean"),
    transformation_decision_accuracy=("transformation_decision_accuracy","mean"),
    false_consolidation_rate=("false_consolidation_rate","mean"),
    anti_core_capture_rate=("anti_core_capture_rate","mean"),
    overall_component_score=("overall_component_score","mean")
).reset_index()
summary.to_csv(BASE/"summary_v261.csv",index=False)

S=summary.set_index("model")
f="full_loop"; ae="ablate_ethicalsense"; ai="ablate_idealgap"; ash="ablate_shadow_validation"; ac="ablate_core_constraint"

# non-target ablation limited check
task_cols = {
    "ethical": ("ethical_detection_accuracy", ae, True),
    "ideal": ("transformation_decision_accuracy", ai, True),
    "shadow": ("false_consolidation_rate", ash, False),
    "core": ("anti_core_capture_rate", ac, False)
}
non_target_ok=True
specificity_ok=True
for _,(col,target,higher_better) in task_cols.items():
    full=S.loc[f,col]
    non_targets=[m for m in [ae,ai,ash,ac] if m!=target]
    within=0
    for m in non_targets:
        if abs(S.loc[m,col]-full)<=0.08:
            within+=1
    if within<2:
        non_target_ok=False

    vals={m:S.loc[m,col] for m in [ae,ai,ash,ac]}
    ordered=sorted(vals.items(), key=lambda kv: kv[1], reverse=not higher_better)
    worst_two=[m for m,_ in ordered[:2]]
    if target not in worst_two:
        specificity_ok=False

checks={
    "ethicalsense_specific_failure": bool(S.loc[ae,"ethical_detection_accuracy"]<=S.loc[f,"ethical_detection_accuracy"]-0.15),
    "idealgap_specific_failure": bool(S.loc[ai,"transformation_decision_accuracy"]<=S.loc[f,"transformation_decision_accuracy"]-0.15),
    "shadow_specific_failure": bool(S.loc[ash,"false_consolidation_rate"]>=S.loc[f,"false_consolidation_rate"]+0.15),
    "core_specific_failure": bool(S.loc[ac,"anti_core_capture_rate"]>=S.loc[f,"anti_core_capture_rate"]+0.20),
    "full_best_overall": bool(all(S.loc[f,"overall_component_score"]>=S.loc[m,"overall_component_score"]+0.08 for m in [ae,ai,ash,ac])),
    "non_target_ablations_limited": bool(non_target_ok),
    "task_specificity": bool(specificity_ok)
}
(BASE/"acceptance_v261.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
