
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v282.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]; IC=SPEC["information_cost"]
models=SPEC["models"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)
def info_cost(q,is_tdc):
    excess=max(0.0,q-IC["free_quality_floor"])
    c=IC["quadratic_cost_scale"]*(excess**2)
    return c*IC["development_discount"] if is_tdc else c

for seed in SPEC["seeds"]:
    for mi,(model,cfg) in enumerate(models.items()):
        rng=np.random.default_rng(seed*100+mi)
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]

        total_traj=0.0; total_info_cost=0.0
        trap_capture=trap_events=0
        final_stats=None

        for t in range(SPEC["episodes"]):
            pi=t//SPEC["phase_length"]
            ph=SPEC["phases"][pi]
            if t % SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            is_tdc=(cfg["type"]=="tdc")
            if not is_tdc:
                q_use=cfg["quality"]; anti=cfg["anti"]
            else:
                mprod=(TF*I*A*(1-H))**0.25
                sig=1/(1+math.exp(-8.0*(mprod-0.58)))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])
                q_use=Q; anti=0.0

            total_info_cost += info_cost(q_use,is_tdc)

            critical=bool(rng.random()<E["critical_prob"])
            irreversible=bool(rng.random()<E["irreversible_given_critical"]) if critical else bool(rng.random()<0.12)
            correct_action=1 if critical else 0
            reward_action=correct_action if rng.random()<ph["reward_alignment"] else 1-correct_action

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal=noisy_bool(rng,critical,norm_acc)
            world_signal=noisy_bool(rng,critical,ph["world_acc"])
            irr_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_acc"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)
            if not is_tdc and reward_action==1:
                score -= anti*(1-ph["reward_alignment"])

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
                if not critical: pst["traj"]-=E["false_alarm_cost"]

            if critical and irreversible:
                pst["crit"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            if ph["trap"]>=0.58 and reward_action!=correct_action:
                trap_events+=1
                if action==reward_action: trap_capture+=1

            if is_tdc:
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
        gross_trajectory=total_traj/SPEC["episodes"]
        avg_info_cost=total_info_cost/SPEC["episodes"]
        net_trajectory=gross_trajectory-avg_info_cost

        rows.append({
            "seed":seed,"model":model,"is_tdc":cfg["type"]=="tdc",
            "final_holdout_recall":final_rec,
            "final_holdout_false_alarm":final_far,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "capture_resistance":1-capture,
            "core_protection":1-final_vr,
            "gross_trajectory":gross_trajectory,
            "avg_information_cost":avg_info_cost,
            "net_trajectory_after_info_cost":net_trajectory,
            "final_quality":final_q
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v282.csv",index=False)

summary=df.groupby(["model","is_tdc"]).agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_false_alarm=("final_holdout_false_alarm","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    capture_resistance=("capture_resistance","mean"),
    core_protection=("core_protection","mean"),
    gross_trajectory=("gross_trajectory","mean"),
    avg_information_cost=("avg_information_cost","mean"),
    net_trajectory_after_info_cost=("net_trajectory_after_info_cost","mean"),
    final_quality=("final_quality","mean")
).reset_index()

objectives=["final_holdout_recall","capture_resistance","core_protection","net_trajectory_after_info_cost"]

def dominates(a,b,eps=1e-12):
    ge=all(float(a[o])>=float(b[o])-eps for o in objectives)
    gt=any(float(a[o])>float(b[o])+eps for o in objectives)
    return bool(ge and gt)

records=summary.set_index("model").to_dict("index")
frontier=[]
for m,a in records.items():
    if not any(n!=m and dominates(b,a) for n,b in records.items()):
        frontier.append(m)

tdc="tdc_evidence_validated"
simple=[m for m in records if m!=tdc]
tdc_dominates=[m for m in simple if dominates(records[tdc],records[m])]
simple_dominates_tdc=[m for m in simple if dominates(records[m],records[tdc])]

summary["pareto_frontier"]=summary["model"].isin(frontier)
summary["dominated_by_tdc"]=summary["model"].isin(tdc_dominates)
summary.to_csv(BASE/"summary_v282.csv",index=False)

simple_df=summary[~summary["is_tdc"]]
strong_high="simple_q090_a016"
wins=sum(float(records[tdc][o])>float(records[strong_high][o]) for o in objectives)

checks={
    "tdc_not_dominated": bool(len(simple_dominates_tdc)==0),
    "tdc_on_pareto_frontier": bool(tdc in frontier),
    "tdc_best_net_trajectory": bool(records[tdc]["net_trajectory_after_info_cost"]>=simple_df["net_trajectory_after_info_cost"].max()-1e-12),
    "tdc_best_or_tied_recall": bool(records[tdc]["final_holdout_recall"]>=simple_df["final_holdout_recall"].max()-0.01),
    "tdc_core_protection_above_all_simple": bool(records[tdc]["core_protection"]>=simple_df["core_protection"].max()-1e-12),
    "complexity_earns_keep_equal_budget": bool(dominates(records[tdc],records[strong_high]) or wins>=3)
}
checks["equal_budget_result_replicates"] = bool(
    checks["tdc_not_dominated"] and checks["tdc_on_pareto_frontier"] and checks["complexity_earns_keep_equal_budget"]
)

(BASE/"acceptance_v282.json").write_text(json.dumps({
    "checks":checks,
    "pareto_frontier":frontier,
    "tdc_dominates":tdc_dominates,
    "simple_dominates_tdc":simple_dominates_tdc,
    "tdc_dominance_fraction":float(len(tdc_dominates)/len(simple)),
    "tdc_wins_vs_q090_a016":int(wins)
},indent=2))

print(summary.to_string(index=False))
print("\nPARETO FRONTIER:",frontier)
print("TDC dominates:",tdc_dominates)
print("Simple policies dominating TDC:",simple_dominates_tdc)
print("TDC wins vs q090/a016:",wins)
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
