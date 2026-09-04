
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v280.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]
simple_specs={x["name"]:x for x in SPEC["simple_policy_family"]}
models=list(simple_specs.keys())+[SPEC["tdc_model"]]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(models):
        rng=np.random.default_rng(seed*100+mi)
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]
        total_traj=0.0
        trap_capture=trap_events=0
        final_stats=None

        for t in range(SPEC["episodes"]):
            pi=t//SPEC["phase_length"]
            ph=SPEC["phases"][pi]
            if t % SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            if model in simple_specs:
                q_use=simple_specs[model]["quality"]
                anti=simple_specs[model]["anti"]
            else:
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])
                q_use=Q
                anti=0.0

            critical=bool(rng.random()<E["critical_prob"])
            irreversible=bool(rng.random()<E["irreversible_given_critical"]) if critical else bool(rng.random()<0.12)
            correct_action=1 if critical else 0
            reward_action=correct_action if rng.random()<ph["reward_alignment"] else 1-correct_action

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal=noisy_bool(rng,critical,norm_acc)
            world_signal=noisy_bool(rng,critical,ph["world_acc"])
            irr_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_acc"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)
            if model in simple_specs and reward_action==1:
                score-=anti*(1-ph["reward_alignment"])

            threshold=0.48 if irr_cue else 0.60

            if irr_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos=noisy_bool(rng,critical,probe_acc)
                score=clip(score+(0.20 if probe_pos else -0.16),0,1)
                pst["traj"]-=E["probe_cost"]

            action=1 if score>=threshold else 0
            trigger=(action==1)

            if trigger:
                if critical and irreversible: pst["tp"]+=1
                elif not critical: pst["fp"]+=1
            else:
                if critical and irreversible: pst["fn"]+=1

            protected=False
            if trigger:
                protected=bool(rng.random()<E["review_success"])
                pst["traj"]-=E["review_cost"]
                if not critical:
                    pst["traj"]-=E["false_alarm_cost"]

            if critical and irreversible:
                pst["crit"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            if ph["trap"]>=0.60 and reward_action!=correct_action:
                trap_events+=1
                if action==reward_action:
                    trap_capture+=1

            if model==SPEC["tdc_model"]:
                validated_success=bool(critical and irreversible and protected)
                normative_correct=(action==correct_action)
                normative_error=bool(critical and irreversible and not protected)

                candidate=(
                    0.40*int(validated_success)
                    +0.28*int(action==reward_action)
                    +0.20*(1-H)
                    +0.12*q_use
                )
                if rng.random()<ph["false_progress"]:
                    candidate+=0.25

                validation=(
                    0.42*int(validated_success)
                    +0.30*int(normative_correct)
                    +0.18*int((action!=reward_action) or reward_action==correct_action)
                    +0.10*q_use
                    +rng.normal(0,0.06)
                )

                if candidate>0.35 and validation>=D["validation_threshold"] and normative_correct:
                    TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                    I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                    A=clip(A+D["eta_A"]*(1-A),0.20,0.95)
                    H=clip(H-D["eta_H"]*H,0.10,0.90)
                    if validated_success:
                        TF=clip(TF+D["validated_gain"]*q_use*(1-TF),0.20,0.95)
                        I=clip(I+0.8*D["validated_gain"]*q_use*(1-I),0.20,0.95)
                elif normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                far=pst["fp"]/pst["n"]
                vr=pst["viol"]/max(1,pst["crit"])
                total_traj+=pst["traj"]
                if pi==len(SPEC["phases"])-1:
                    final_stats=(rec,far,vr,q_use)

        final_rec,final_far,final_vr,final_q=final_stats
        capture=trap_capture/max(1,trap_events)
        trajectory=total_traj/SPEC["episodes"]

        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":final_rec,
            "final_holdout_false_alarm":final_far,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "capture_resistance":1-capture,
            "core_protection":1-final_vr,
            "trajectory_value":trajectory,
            "final_quality":final_q,
            "is_tdc": model==SPEC["tdc_model"]
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v280.csv",index=False)

summary=df.groupby(["model","is_tdc"]).agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_false_alarm=("final_holdout_false_alarm","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    capture_resistance=("capture_resistance","mean"),
    core_protection=("core_protection","mean"),
    trajectory_value=("trajectory_value","mean"),
    final_quality=("final_quality","mean")
).reset_index()

objectives=["final_holdout_recall","trajectory_value","capture_resistance","core_protection"]

def dominates(a,b,eps=1e-12):
    ge=all(a[o]>=b[o]-eps for o in objectives)
    gt=any(a[o]>b[o]+eps for o in objectives)
    return ge and gt

records=summary.set_index("model").to_dict("index")
frontier=[]
for m,a in records.items():
    dominated=False
    for n,b in records.items():
        if m!=n and dominates(b,a):
            dominated=True
            break
    if not dominated:
        frontier.append(m)

tdc=SPEC["tdc_model"]
simple=[m for m in records if m!=tdc]
tdc_dominates=[m for m in simple if dominates(records[tdc],records[m])]
simple_dominates_tdc=[m for m in simple if dominates(records[m],records[tdc])]

summary["pareto_frontier"]=summary["model"].isin(frontier)
summary["dominated_by_tdc"]=summary["model"].isin(tdc_dominates)
summary.to_csv(BASE/"summary_v280.csv",index=False)

simple_df=summary[~summary["is_tdc"]]
checks={
    "tdc_not_dominated": len(simple_dominates_tdc)==0,
    "tdc_dominates_at_least_half": len(tdc_dominates)>=len(simple)/2,
    "tdc_on_global_pareto_frontier": tdc in frontier,
    "tdc_best_or_tied_recall": records[tdc]["final_holdout_recall"]>=simple_df["final_holdout_recall"].max()-0.01,
    "tdc_best_or_tied_trajectory": records[tdc]["trajectory_value"]>=simple_df["trajectory_value"].max()-0.01,
    "tdc_capture_resistance_above_family_median": records[tdc]["capture_resistance"]>simple_df["capture_resistance"].median(),
    "tdc_core_protection_above_family_median": records[tdc]["core_protection"]>simple_df["core_protection"].median()
}

(BASE/"acceptance_v280.json").write_text(json.dumps({
    "checks":checks,
    "pareto_frontier":frontier,
    "tdc_dominates":tdc_dominates,
    "simple_dominates_tdc":simple_dominates_tdc,
    "tdc_dominance_fraction":len(tdc_dominates)/len(simple)
},indent=2))

print(summary.to_string(index=False))
print("\nPARETO FRONTIER:", frontier)
print("TDC dominates:", tdc_dominates)
print("Simple policies dominating TDC:", simple_dominates_tdc)
print("TDC dominance fraction:", len(tdc_dominates)/len(simple))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
