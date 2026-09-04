
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v265.json").read_text())
E=SPEC["environment"]
R=SPEC["review"]
INIT=SPEC["initial_state"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng, truth, acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)
        C=INIT["core_detection_reliability"]
        W=INIT["world_model_skill"]
        H=INIT["H_self"]

        critical_n=0
        critical_ok=0
        irreversible_core_viol=0
        review_steps=0
        reversible_choices=0
        reversible_available_n=0
        traj=0.0

        for _ in range(SPEC["episodes"]):
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_action_probability"])
            pressure=bool(rng.random()<E["external_pressure_probability"])
            safe_reversible_available=bool(rng.random()<E["safe_reversible_option_probability"])

            # latent truth: should block the risky action in critical cases
            should_block=critical
            core_signal=noisy_bool(rng, should_block, E["core_signal_accuracy"])
            world_signal=noisy_bool(rng, should_block, E["world_signal_accuracy"])

            if pressure:
                pressure_pushes_risk = bool(rng.random()<E["pressure_wrong_probability"])
            else:
                pressure_pushes_risk=False

            perceived_risk_score = 0.50*int(core_signal) + 0.35*int(world_signal) + 0.15*int(pressure_pushes_risk)
            action_block = perceived_risk_score >= 0.50

            if critical:
                critical_n += 1

                if model=="brute_critical_rumination":
                    # replay same representation; little epistemic benefit, rising H_self
                    for c in range(8):
                        review_steps += 1
                        H=clip(H+0.012*(0.7+0.05*c),0.10,0.95)
                        # slight reinforcement of existing belief, correct or not
                        perceived_risk_score=clip(perceived_risk_score + 0.015*(perceived_risk_score-0.5),0,1)
                    action_block = perceived_risk_score >= 0.50

                elif model=="structured_critical_review":
                    votes=[]
                    found_reversible=False
                    for c in range(R["max_cycles"]):
                        review_steps += 1
                        # independent noisy checks of different failure angles
                        check_acc=clip(0.68+0.025*c+0.08*C+0.05*W,0.55,0.90)
                        vote=noisy_bool(rng, should_block, check_acc)
                        votes.append(1 if vote else 0)

                        if safe_reversible_available and rng.random() < (0.28+0.06*c):
                            found_reversible=True

                        consensus=sum(votes)/len(votes)
                        if len(votes)>=3 and (consensus>=R["stop_consensus"] or consensus<=(1-R["stop_consensus"])):
                            break
                        if found_reversible and len(votes)>=2:
                            break

                    if found_reversible:
                        action_block=True
                        reversible_choices+=1
                    else:
                        action_block=(sum(votes)/len(votes))>=0.5

                    H=clip(H-0.010*(len(votes)/R["max_cycles"]),0.10,0.95)

            # choice evaluation
            correct = (action_block == should_block)
            if critical:
                critical_ok += int(correct)
                if safe_reversible_available:
                    reversible_available_n += 1

            if critical and irreversible and not action_block:
                irreversible_core_viol += 1
                traj -= E["severe_core_violation_penalty"]
            elif correct:
                traj += 1.0
            else:
                traj -= E["normal_error_penalty"]

            # review computation cost
            # applied incrementally as average cost to trajectory
            # one step = small opportunity cost

        traj -= review_steps * E["review_step_cost"]

        rows.append({
            "seed":seed,
            "model":model,
            "irreversible_core_violation_rate":irreversible_core_viol/max(1,critical_n),
            "critical_scenario_accuracy":critical_ok/max(1,critical_n),
            "review_steps_per_critical_event":review_steps/max(1,critical_n),
            "safe_reversible_choice_rate":reversible_choices/max(1,reversible_available_n),
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "final_H_self":H
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v265.csv",index=False)

summary=df.groupby("model").agg(
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    critical_scenario_accuracy=("critical_scenario_accuracy","mean"),
    review_steps_per_critical_event=("review_steps_per_critical_event","mean"),
    safe_reversible_choice_rate=("safe_reversible_choice_rate","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    final_H_self=("final_H_self","mean")
).reset_index()
summary.to_csv(BASE/"summary_v265.csv",index=False)

S=summary.set_index("model")
s="structured_critical_review"
n="no_emergency_loop"
b="brute_critical_rumination"
best_nb=max(S.loc[n,"long_term_trajectory_value"],S.loc[b,"long_term_trajectory_value"])

checks={
    "structured_reduces_core_violations_vs_none": bool(S.loc[s,"irreversible_core_violation_rate"]<=S.loc[n,"irreversible_core_violation_rate"]-0.08),
    "structured_reduces_core_violations_vs_brute": bool(S.loc[s,"irreversible_core_violation_rate"]<=S.loc[b,"irreversible_core_violation_rate"]-0.05),
    "structured_improves_critical_accuracy": bool(S.loc[s,"critical_scenario_accuracy"]>=S.loc[n,"critical_scenario_accuracy"]+0.08),
    "structured_uses_reversible_options": bool(S.loc[s,"safe_reversible_choice_rate"]>=S.loc[n,"safe_reversible_choice_rate"]+0.10),
    "structured_longterm_not_worse": bool(S.loc[s,"long_term_trajectory_value"]>=best_nb-0.01),
    "brute_not_best": bool(S.loc[b,"irreversible_core_violation_rate"]>S.loc[s,"irreversible_core_violation_rate"]),
    "brute_increases_Hself": bool(S.loc[b,"final_H_self"]>=S.loc[s,"final_H_self"]+0.08)
}
(BASE/"acceptance_v265.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
