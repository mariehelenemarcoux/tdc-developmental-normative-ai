
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v266.json").read_text())
E=SPEC["environment"]
R=SPEC["review"]
P=SPEC["developmental_parameters"]
INIT=SPEC["initial_state"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)
        Re=INIT["ethical_reliability"]
        TF=INIT["third_factor_volition"]
        I=INIT["integration"]
        Align=INIT["ideal_alignment"]
        H=INIT["H_self"]
        W=INIT["world_confidence"]

        p2_n=p2_ok=0
        p2_capture=0
        critical_n=0
        irreversible_viol=0
        abstain_n=0
        review_triggers=0
        review_steps=0
        reversible_choices=0
        reversible_available_n=0
        probes=0
        traj=0.0

        for t in range(SPEC["episodes"]):
            phase=1 if t<SPEC["phase1_episodes"] else 2
            reward_align_prob=E["phase1_reward_alignment_probability"] if phase==1 else E["phase2_reward_alignment_probability"]

            fact=1 if rng.random()<0.5 else -1
            core=1 if rng.random()<0.5 else -1
            long=1 if rng.random()<0.5 else -1

            latent_score=0.45*fact+0.92*core+0.80*long+0.42*core*long
            y=sign(latent_score)

            fact_obs=noisy(rng,fact,E["world_signal_accuracy"])
            core_obs=noisy(rng,core,E["normative_signal_accuracy"])
            long_obs=noisy(rng,long,E["normative_signal_accuracy"])

            high=bool(rng.random()<E["high_stakes_probability"])
            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_action_probability"])
            safe_reversible_available=bool(rng.random()<E["safe_reversible_option_probability"])

            pressure=bool(rng.random()<E["external_pressure_probability"])
            pressure_action=(-y if rng.random()<E["pressure_wrong_probability"] else y) if pressure else None
            reward_action=y if rng.random()<reward_align_prob else -y

            world_est=fact_obs if rng.random()<W else -fact_obs
            norm_ground=sign(0.60*core+0.40*long)
            norm_obs=sign(0.60*core_obs+0.40*long_obs)
            ethical_interp=norm_obs if rng.random()<Re else -norm_obs
            internal_action=sign(0.34*world_est+0.66*ethical_interp)

            conflict=bool(pressure and pressure_action!=internal_action)
            core_long_disagree=(core_obs!=long_obs)
            ethical_salience=clip(
                0.30*(1 if conflict else 0)
                +0.30*(1 if high else 0)
                +0.40*(1 if core_long_disagree else 0),0,1
            )

            core_risk_score=clip(
                0.28*(1 if critical else 0)
                +0.22*(1 if irreversible else 0)
                +0.20*(1 if conflict else 0)
                +0.18*(1 if core_long_disagree else 0)
                +0.12*(1-Re),
                0,1
            )

            authority=clip(0.40*TF+0.22*Re+0.16*I+0.12*Align+0.10*(1-H),0,1)
            independent=authority>=0.30
            proposed_action=pressure_action if (pressure and not independent) else internal_action

            abstain=False
            action=proposed_action

            # --- emergency review ---
            trigger_threshold = R["normal_trigger_threshold"]
            if model=="full_hypersensitive_emergency":
                trigger_threshold=R["hypersensitive_trigger_threshold"]

            do_review = model!="full_no_emergency" and core_risk_score>=trigger_threshold

            if do_review:
                review_triggers+=1
                votes=[]
                found_reversible=False
                used_probe=False

                # VOI: if uncertain and reversible info possible, probe first
                if model=="full_emergency_voi":
                    uncertainty=1-abs(core_risk_score-0.5)*2
                    voi=uncertainty*(1 if irreversible else 0.6)*(1 if critical else 0.7)
                    if voi>0.28:
                        probes+=1
                        used_probe=True
                        # noisy extra evidence about whether proposed action is dangerous
                        dangerous = (critical and proposed_action!=y)
                        probe=noisy_bool(rng,dangerous,E["probe_success_probability"])
                        if probe:
                            core_risk_score=clip(core_risk_score+0.18,0,1)
                        else:
                            core_risk_score=clip(core_risk_score-0.12,0,1)

                for c in range(R["max_cycles"]):
                    review_steps+=1
                    dangerous = (critical and proposed_action!=y)
                    check_acc=clip(0.66+0.025*c+0.08*Re+0.05*W,0.55,0.90)
                    vote=noisy_bool(rng,dangerous,check_acc)
                    votes.append(1 if vote else 0)

                    if safe_reversible_available and rng.random()<(0.24+0.06*c):
                        found_reversible=True

                    consensus=sum(votes)/len(votes)
                    if len(votes)>=3 and (consensus>=R["consensus_threshold"] or consensus<=1-R["consensus_threshold"]):
                        break
                    if found_reversible and len(votes)>=2:
                        break

                if found_reversible:
                    action = y if rng.random()<0.84 else -y  # reversible/safe option is usually Core-consistent
                    reversible_choices+=1
                    if critical:
                        reversible_available_n+=1
                else:
                    danger_consensus=sum(votes)/len(votes)
                    if danger_consensus>=0.5:
                        # if irreversible and danger remains unresolved, abstain/defer
                        if irreversible:
                            abstain=True
                            abstain_n+=1
                        else:
                            action=-proposed_action

                H=clip(H-0.008*(len(votes)/R["max_cycles"]),0.10,0.95)

            # action evaluation
            if critical:
                critical_n+=1

            if abstain:
                # deferral preserves Core but has opportunity cost
                traj-=0.10
                ok=False
            else:
                ok=(action==y)
                if phase==2:
                    p2_n+=1
                    p2_ok+=int(ok)
                    if action==reward_action and reward_action!=y:
                        p2_capture+=1

                if critical and irreversible and not ok:
                    irreversible_viol+=1
                    traj-=E["severe_core_violation_penalty"]
                elif ok:
                    traj+=1.0
                else:
                    traj-=E["normal_error_penalty"]

            # development loop after outcome
            ideal_gap=max(0.0,1.0-Align)
            drive=ethical_salience*Re*ideal_gap
            normative_correct=(ethical_interp==norm_ground)

            if drive>P["candidate_threshold"]:
                candidate_wrong=bool(rng.random()<E["seductive_wrong_candidate_probability"])
                candidate_correct=not candidate_wrong

                if not candidate_wrong:
                    candidate_correct = rng.random()<clip(0.52+0.24*Re+0.18*I,0.50,0.95)

                accepted = candidate_correct if rng.random()<E["shadow_validation_accuracy"] else (not candidate_correct)
                if accepted:
                    TF=clip(TF+P["eta_TF"]*drive,INIT["third_factor_volition"],0.95)
                    if candidate_correct:
                        I=clip(I+P["eta_I"]*drive,0.10,0.95)
                        Align=clip(Align+P["eta_align"]*drive,0.10,0.95)
                        Re=clip(Re+P["eta_R"]*I*drive,0.50,0.94)
                        H=clip(H-P["eta_H_down"]*drive,0.10,0.95)
                    else:
                        H=clip(H+P["eta_H_up"]*drive,0.10,0.95)

            if ethical_salience>0.35:
                Re=clip(Re+(0.0012*I if normative_correct else -0.0010),0.50,0.94)

        effective_p2_n=max(1,p2_n)
        p2_acc=p2_ok/effective_p2_n
        cap=p2_capture/effective_p2_n
        viol=irreversible_viol/max(1,critical_n)
        abst=abstain_n/SPEC["episodes"]
        rev_rate=reversible_choices/max(1,reversible_available_n)
        review_rate=review_triggers/SPEC["episodes"]
        steps_per_episode=review_steps/SPEC["episodes"]
        probe_rate=probes/SPEC["episodes"]

        developmental_transfer=(
            0.30*p2_acc+0.25*Align+0.20*I+0.15*(1-H)+0.10*(1-cap)
        )

        integrated=(
            0.22*p2_acc+0.18*Align+0.14*I+0.14*(1-H)
            +0.12*(1-cap)+0.10*(1-viol)+0.10*(1-abst)
        )

        # computation costs
        traj -= review_steps*E["review_step_cost"]
        traj -= probes*E["probe_cost"]

        rows.append({
            "seed":seed,"model":model,
            "irreversible_core_violation_rate":viol,
            "phase2_core_accuracy":p2_acc,
            "abstention_rate":abst,
            "review_trigger_rate":review_rate,
            "review_steps_per_episode":steps_per_episode,
            "safe_reversible_choice_rate":rev_rate,
            "probe_rate":probe_rate,
            "phase2_reward_capture_rate":cap,
            "developmental_transfer_score":developmental_transfer,
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "integrated_score":integrated,
            "final_H_self":H,
            "final_ethical_reliability":Re,
            "final_integration":I,
            "final_ideal_alignment":Align
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v266.csv",index=False)

summary=df.groupby("model").agg(
    irreversible_core_violation_rate=("irreversible_core_violation_rate","mean"),
    phase2_core_accuracy=("phase2_core_accuracy","mean"),
    abstention_rate=("abstention_rate","mean"),
    review_trigger_rate=("review_trigger_rate","mean"),
    review_steps_per_episode=("review_steps_per_episode","mean"),
    safe_reversible_choice_rate=("safe_reversible_choice_rate","mean"),
    probe_rate=("probe_rate","mean"),
    phase2_reward_capture_rate=("phase2_reward_capture_rate","mean"),
    developmental_transfer_score=("developmental_transfer_score","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    integrated_score=("integrated_score","mean"),
    final_H_self=("final_H_self","mean"),
    final_ethical_reliability=("final_ethical_reliability","mean"),
    final_integration=("final_integration","mean"),
    final_ideal_alignment=("final_ideal_alignment","mean")
).reset_index()
summary.to_csv(BASE/"summary_v266.csv",index=False)

S=summary.set_index("model")
n="full_no_emergency"
e="full_core_emergency"
h="full_hypersensitive_emergency"
v="full_emergency_voi"

checks={
    "emergency_reduces_core_violations": bool(S.loc[e,"irreversible_core_violation_rate"]<=S.loc[n,"irreversible_core_violation_rate"]-0.05),
    "emergency_no_chronic_abstention": bool(S.loc[e,"abstention_rate"]<=0.18),
    "emergency_integrated_score_not_worse": bool(S.loc[e,"integrated_score"]>=S.loc[n,"integrated_score"]-0.01),
    "hypersensitive_more_abstention": bool(S.loc[h,"abstention_rate"]>=S.loc[e,"abstention_rate"]+0.06),
    "voi_best_or_tied_core_protection": bool(S.loc[v,"irreversible_core_violation_rate"]<=S.loc[e,"irreversible_core_violation_rate"]+0.01),
    "voi_more_efficient": bool(
        S.loc[v,"review_steps_per_episode"]<=S.loc[e,"review_steps_per_episode"]
        or S.loc[v,"long_term_trajectory_value"]>=S.loc[e,"long_term_trajectory_value"]+0.02
    ),
    "voi_best_integrated_score": bool(
        S.loc[v,"integrated_score"]>=S["integrated_score"].max()-1e-12
    )
}
(BASE/"acceptance_v266.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,vv in checks.items():
    print(f"{k}: {vv}")
print(f"passed={sum(checks.values())}/{len(checks)}")
