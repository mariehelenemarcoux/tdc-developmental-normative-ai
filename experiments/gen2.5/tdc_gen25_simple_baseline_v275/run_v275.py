
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v275.json").read_text())
D=SPEC["development"]; B=SPEC["simple_baselines"]; E=SPEC["environment"]
rows=[]; phase_rows=[]

def clip(x,a,b): return max(a,min(b,x))
def noisy_bool(rng,truth,acc): return truth if rng.random()<acc else (not truth)

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)

        TF=SPEC["initial_state"]["third_factor_volition"]
        I=SPEC["initial_state"]["integration"]
        A=SPEC["initial_state"]["ideal_alignment"]
        H=SPEC["initial_state"]["H_self"]
        Q=SPEC["initial_state"]["ethical_signal_quality"]
        Q_online=B["online_initial_quality"]

        phase_stats=[]
        total_traj=0.0

        for t in range(SPEC["episodes"]):
            phase_index=t//SPEC["phase_length"]+1
            if t % SPEC["phase_length"]==0:
                pst={"tp_i":0,"fn_i":0,"fp":0,"crit_i":0,"viol":0,"n":0,"traj":0.0}

            # Signal quality used by each model.
            if model=="tdc_feedback_development":
                maturity_prod=(TF*I*A*(1-H))**0.25
                x=8.0*(maturity_prod-0.58)
                sig=1/(1+math.exp(-x))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])
                q_use=Q
            elif model=="simple_static_threshold":
                q_use=B["static_quality"]
            elif model=="simple_quality_schedule":
                frac=t/max(1,SPEC["episodes"]-1)
                q_use=B["schedule_start"]+(B["schedule_end"]-B["schedule_start"])*frac
            else:
                q_use=Q_online

            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_given_critical_probability"]) if critical else bool(rng.random()<0.12)

            core_signal=noisy_bool(rng,critical,q_use)
            world_signal=noisy_bool(rng,critical,E["base_world_signal_accuracy"])
            irreversible_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_accuracy"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irreversible_cue)
            threshold=0.48 if irreversible_cue else 0.60

            if irreversible_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos=noisy_bool(rng,critical,probe_acc)
                score=clip(score+(0.20 if probe_pos else -0.16),0,1)
                pst["traj"]-=E["probe_cost"]

            trigger=score>=threshold
            if trigger:
                if critical and irreversible: pst["tp_i"]+=1
                elif not critical: pst["fp"]+=1
            else:
                if critical and irreversible: pst["fn_i"]+=1

            protected=False
            if trigger:
                protected=bool(rng.random()<E["review_success_probability"])
                pst["traj"]-=E["review_cost"]
                if not critical: pst["traj"]-=E["false_alarm_cost"]

            if critical and irreversible:
                pst["crit_i"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            validated_success=bool(critical and irreversible and protected)
            normative_error=bool(critical and irreversible and not protected)

            # Updates.
            if model=="tdc_feedback_development":
                if validated_success:
                    TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                    I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                    A=clip(A+D["eta_align"]*(1-A),0.20,0.95)
                    H=clip(H-D["eta_H"]*H,0.10,0.90)
                    TF=clip(TF+D["validated_action_gain"]*q_use*(1-TF),0.20,0.95)
                    I=clip(I+0.8*D["validated_action_gain"]*q_use*(1-I),0.20,0.95)
                elif normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

            elif model=="simple_online_calibration":
                if validated_success:
                    Q_online=clip(Q_online+B["online_eta_up"]*(1-Q_online),0.50,0.94)
                elif normative_error:
                    Q_online=clip(Q_online-B["online_eta_down"]*Q_online,0.50,0.94)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp_i"]/max(1,pst["tp_i"]+pst["fn_i"])
                far=pst["fp"]/pst["n"]
                vr=pst["viol"]/max(1,pst["crit_i"])
                traj=pst["traj"]/pst["n"]
                phase_rows.append({
                    "seed":seed,"model":model,"phase":phase_index,
                    "signal_quality":q_use,
                    "irreversible_recall":rec,
                    "false_alarm_rate":far,
                    "violation_rate":vr,
                    "trajectory":traj
                })
                phase_stats.append((rec,far,vr,traj,q_use))
                total_traj += pst["traj"]

        final_rec,final_far,final_vr,final_phase_traj,final_q=phase_stats[-1]
        net=0.45*final_rec+0.20*(1-final_far)+0.20*(1-final_vr)+0.15*final_q

        state_complexity = 4 if model=="tdc_feedback_development" else (1 if model=="simple_online_calibration" else 0)

        rows.append({
            "seed":seed,"model":model,
            "final_signal_quality":final_q,
            "irreversible_recall_final_phase":final_rec,
            "false_alarm_rate_final_phase":final_far,
            "irreversible_core_violation_rate_final_phase":final_vr,
            "net_vigilance_value_final_phase":net,
            "long_term_trajectory_value":total_traj/SPEC["episodes"],
            "state_complexity_count":state_complexity
        })

df=pd.DataFrame(rows); ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v275.csv",index=False)
ph.to_csv(BASE/"phase_results_v275.csv",index=False)

summary=df.groupby("model").agg(
    final_signal_quality=("final_signal_quality","mean"),
    irreversible_recall_final_phase=("irreversible_recall_final_phase","mean"),
    false_alarm_rate_final_phase=("false_alarm_rate_final_phase","mean"),
    irreversible_core_violation_rate_final_phase=("irreversible_core_violation_rate_final_phase","mean"),
    net_vigilance_value_final_phase=("net_vigilance_value_final_phase","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    state_complexity_count=("state_complexity_count","mean")
).reset_index()
summary.to_csv(BASE/"summary_v275.csv",index=False)

S=summary.set_index("model")
t="tdc_feedback_development"
st="simple_static_threshold"
sc="simple_quality_schedule"
on="simple_online_calibration"

strongest_simple = S.loc[[st,sc,on],"net_vigilance_value_final_phase"].idxmax()
strongest_simple_net = S.loc[strongest_simple,"net_vigilance_value_final_phase"]
strongest_margin = S.loc[t,"net_vigilance_value_final_phase"] - strongest_simple_net

checks={
    "tdc_beats_static_recall": bool(S.loc[t,"irreversible_recall_final_phase"]>=S.loc[st,"irreversible_recall_final_phase"]+0.12),
    "tdc_beats_static_net": bool(S.loc[t,"net_vigilance_value_final_phase"]>=S.loc[st,"net_vigilance_value_final_phase"]+0.10),
    "tdc_beats_schedule_net": bool(S.loc[t,"net_vigilance_value_final_phase"]>=S.loc[sc,"net_vigilance_value_final_phase"]+0.03),
    "tdc_beats_online_net": bool(S.loc[t,"net_vigilance_value_final_phase"]>=S.loc[on,"net_vigilance_value_final_phase"]+0.03),
    "tdc_lower_violations_than_all_simple": bool(
        S.loc[t,"irreversible_core_violation_rate_final_phase"]<=
        S.loc[[st,sc,on],"irreversible_core_violation_rate_final_phase"].min()
    ),
    "tdc_best_trajectory": bool(
        S.loc[t,"long_term_trajectory_value"]>=S.loc[[st,sc,on],"long_term_trajectory_value"].max()
    ),
    "complexity_earns_keep": bool(
        S.loc[t,"net_vigilance_value_final_phase"]>=strongest_simple_net and strongest_margin>=0.03
    )
}
(BASE/"acceptance_v275.json").write_text(json.dumps({
    "strongest_simple_baseline": strongest_simple,
    "strongest_simple_margin": float(strongest_margin),
    "checks": checks
},indent=2))

print(summary.to_string(index=False))
print("\nSTRONGEST SIMPLE BASELINE:", strongest_simple)
print("TDC NET MARGIN:", strongest_margin)
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
