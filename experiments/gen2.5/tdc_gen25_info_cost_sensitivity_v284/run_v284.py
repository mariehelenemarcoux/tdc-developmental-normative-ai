
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v284.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]; IC=SPEC["information_cost"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)
def base_info_cost(q):
    excess=max(0.0,q-IC["free_quality_floor"])
    return IC["quadratic_cost_scale"]*(excess**2)

for seed in SPEC["seeds"]:
    # Freeze one behavioral trajectory per model/seed; discounts applied only after.
    behavioral={}
    for mi,(model,cfg) in enumerate(SPEC["models"].items()):
        rng=np.random.default_rng(seed*100+mi)
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        A=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]

        total_traj=0.0
        total_base_info=0.0
        trap_capture=trap_events=0
        final_stats=None

        for t in range(SPEC["episodes"]):
            pi=t//SPEC["phase_length"]; ph=SPEC["phases"][pi]
            if t%SPEC["phase_length"]==0:
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

            total_base_info += base_info_cost(q_use)

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

                candidate=(0.40*int(validated_success)+0.28*int(action==reward_action)+0.20*(1-H)+0.12*q_use)
                if rng.random()<ph["false_progress"]: candidate+=0.25

                validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                            +0.18*int((action!=reward_action) or reward_action==correct_action)
                            +0.10*q_use+rng.normal(0,0.06))

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
                vr=pst["viol"]/max(1,pst["crit"])
                total_traj+=pst["traj"]
                if pi==len(SPEC["phases"])-1:
                    final_stats=(rec,vr,q_use)

        final_rec,final_vr,final_q=final_stats
        behavioral[model]={
            "gross_trajectory":total_traj/SPEC["episodes"],
            "avg_base_info_cost":total_base_info/SPEC["episodes"],
            "final_recall":final_rec,
            "core_protection":1-final_vr,
            "capture_resistance":1-trap_capture/max(1,trap_events),
            "final_quality":final_q
        }

    for discount in SPEC["discount_levels"]:
        for model,b in behavioral.items():
            is_tdc=(SPEC["models"][model]["type"]=="tdc")
            applied_discount=discount if is_tdc else 1.0
            net=b["gross_trajectory"]-applied_discount*b["avg_base_info_cost"]
            rows.append({
                "seed":seed,"discount":discount,"model":model,
                "gross_trajectory":b["gross_trajectory"],
                "avg_base_info_cost":b["avg_base_info_cost"],
                "applied_discount":applied_discount,
                "net_trajectory":net,
                "final_recall":b["final_recall"],
                "core_protection":b["core_protection"],
                "capture_resistance":b["capture_resistance"],
                "final_quality":b["final_quality"]
            })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v284.csv",index=False)

summary=df.groupby(["discount","model"]).agg(
    gross_trajectory=("gross_trajectory","mean"),
    avg_base_info_cost=("avg_base_info_cost","mean"),
    net_trajectory=("net_trajectory","mean"),
    final_recall=("final_recall","mean"),
    core_protection=("core_protection","mean"),
    capture_resistance=("capture_resistance","mean"),
    final_quality=("final_quality","mean")
).reset_index()
summary.to_csv(BASE/"summary_v284.csv",index=False)

curve=[]
for d in SPEC["discount_levels"]:
    sub=summary[summary["discount"]==d].set_index("model")
    tdc=sub.loc["tdc_evidence_validated"]
    simple_names=["simple_q090_a008","simple_q090_a016"]
    best_name=sub.loc[simple_names,"net_trajectory"].idxmax()
    best=sub.loc[best_name]
    curve.append({
        "discount":d,
        "tdc_net_trajectory":float(tdc["net_trajectory"]),
        "best_simple_net_trajectory":float(best["net_trajectory"]),
        "best_simple_name":best_name,
        "tdc_margin_vs_best_simple":float(tdc["net_trajectory"]-best["net_trajectory"]),
        "tdc_info_cost_before_discount":float(tdc["avg_base_info_cost"]),
        "best_simple_info_cost":float(best["avg_base_info_cost"])
    })

curve_df=pd.DataFrame(curve)
curve_df.to_csv(BASE/"break_even_curve_v284.csv",index=False)

# Linear interpolation between adjacent sign-changing points.
be=None
for i in range(len(curve)-1):
    x1,y1=curve[i]["discount"],curve[i]["tdc_margin_vs_best_simple"]
    x2,y2=curve[i+1]["discount"],curve[i+1]["tdc_margin_vs_best_simple"]
    if y1==0:
        be=x1; break
    if y1*y2<0:
        be=x1+(0-y1)*(x2-x1)/(y2-y1)
        break

margins=curve_df["tdc_margin_vs_best_simple"].tolist()
mono=all(margins[i+1]<=margins[i]+1e-12 for i in range(len(margins)-1))

# Behavior invariance
behavior_cols=["final_recall","core_protection","capture_resistance","final_quality","gross_trajectory"]
max_spread=0.0
for model in summary["model"].unique():
    sm=summary[summary["model"]==model]
    for c in behavior_cols:
        max_spread=max(max_spread,float(sm[c].max()-sm[c].min()))

def margin_at(d):
    return float(curve_df.loc[np.isclose(curve_df["discount"],d),"tdc_margin_vs_best_simple"].iloc[0])

checks={
    "margin_decreases_monotonically_with_discount": bool(mono),
    "positive_margin_at_0_25": bool(margin_at(0.25)>0),
    "positive_margin_at_0_50": bool(margin_at(0.50)>0),
    "negative_margin_at_1_00": bool(margin_at(1.00)<0),
    "break_even_exists": bool(be is not None),
    "break_even_between_0_50_and_1_00": bool(be is not None and 0.50<=be<=1.00),
    "behavior_metrics_invariant_across_discounts": bool(max_spread<1e-12)
}

(BASE/"acceptance_v284.json").write_text(json.dumps({
    "checks":checks,
    "estimated_break_even_discount":None if be is None else float(be),
    "max_behavior_metric_spread":float(max_spread)
},indent=2))

print(curve_df.to_string(index=False))
print("\nEstimated break-even discount:",be)
print("Max behavior metric spread:",max_spread)
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
