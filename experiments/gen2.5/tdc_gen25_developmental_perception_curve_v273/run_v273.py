
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v273.json").read_text())
D=SPEC["development"]
E=SPEC["environment"]
rows=[]
phase_rows=[]

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

        phase_index=0
        crossing_phase=None
        phase_stats=[]

        for t in range(SPEC["episodes"]):
            if t % SPEC["phase_length"] == 0:
                phase_index += 1
                pst={"tp_i":0,"fn_i":0,"fp":0,"crit_i":0,"viol":0,"n":0,"traj":0.0}

            maturity_mean=(TF+I+A+(1-H))/4.0
            maturity_prod=(TF*I*A*(1-H))**0.25

            if model=="static_quality":
                Q=SPEC["initial_state"]["ethical_signal_quality"]
            elif model=="linear_development":
                Q=clip(D["quality_floor"] + (D["quality_ceiling"]-D["quality_floor"])*maturity_mean,
                       D["quality_floor"],D["quality_ceiling"])
            elif model in ("interaction_development","feedback_development"):
                # nonlinear interaction, with steeper gain after joint maturity ~0.60
                x=8.0*(maturity_prod-0.58)
                sig=1/(1+math.exp(-x))
                Q=clip(D["quality_floor"]+(D["quality_ceiling"]-D["quality_floor"])*sig,
                       D["quality_floor"],D["quality_ceiling"])

            if crossing_phase is None and Q>=SPEC["threshold_reference"]:
                crossing_phase=phase_index

            critical=bool(rng.random()<E["critical_core_risk_probability"])
            irreversible=bool(rng.random()<E["irreversible_given_critical_probability"]) if critical else bool(rng.random()<0.10)

            core_signal=noisy_bool(rng,critical,Q)
            world_signal=noisy_bool(rng,critical,E["base_world_signal_accuracy"])
            irreversible_cue=noisy_bool(rng,irreversible,E["irreversibility_cue_accuracy"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irreversible_cue)
            threshold=0.48 if irreversible_cue else 0.60

            # targeted boundary probe
            if irreversible_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(Q+E["probe_bonus"],0.55,0.98)
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
                if not critical:
                    pst["traj"]-=E["false_alarm_cost"]

            if critical and irreversible:
                pst["crit_i"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            # post-action developmental updates
            validated_success = bool(critical and irreversible and protected)
            normative_error = bool(critical and irreversible and not protected)

            if model!="static_quality":
                # third factor/integration evolve from successful internally guided action
                if validated_success:
                    TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                    I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                    A=clip(A+D["eta_align"]*(1-A),0.20,0.95)
                    H=clip(H-D["eta_H"]*H,0.10,0.90)
                elif normative_error:
                    H=clip(H+D["error_penalty"]*(1-H),0.10,0.90)

                if model=="feedback_development" and validated_success:
                    TF=clip(TF+D["validated_action_gain"]*Q*(1-TF),0.20,0.95)
                    I=clip(I+0.8*D["validated_action_gain"]*Q*(1-I),0.20,0.95)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp_i"]/max(1,pst["tp_i"]+pst["fn_i"])
                far=pst["fp"]/pst["n"]
                vr=pst["viol"]/max(1,pst["crit_i"])
                phase_rows.append({
                    "seed":seed,"model":model,"phase":phase_index,
                    "Q":Q,"TF":TF,"I":I,"A":A,"H":H,
                    "irreversible_recall":rec,
                    "false_alarm_rate":far,
                    "violation_rate":vr,
                    "trajectory":pst["traj"]/pst["n"]
                })
                phase_stats.append((rec,far,vr,pst["traj"]/pst["n"]))

        final_rec,final_far,final_vr,final_traj=phase_stats[-1]
        maturity_final=(TF+I+A+(1-H))/4.0
        rows.append({
            "seed":seed,"model":model,
            "final_ethical_signal_quality":Q,
            "phase_crossing_index": crossing_phase if crossing_phase is not None else 99,
            "irreversible_recall_final_phase":final_rec,
            "false_alarm_rate_final_phase":final_far,
            "irreversible_core_violation_rate_final_phase":final_vr,
            "net_vigilance_value_final_phase": (
                0.45*final_rec+0.20*(1-final_far)+0.20*(1-final_vr)+0.15*Q
            ),
            "developmental_maturity_final":maturity_final,
            "final_TF":TF,"final_I":I,"final_A":A,"final_H":H
        })

df=pd.DataFrame(rows)
ph=pd.DataFrame(phase_rows)
df.to_csv(BASE/"seed_results_v273.csv",index=False)
ph.to_csv(BASE/"phase_results_v273.csv",index=False)

summary=df.groupby("model").agg(
    final_ethical_signal_quality=("final_ethical_signal_quality","mean"),
    phase_crossing_index=("phase_crossing_index","mean"),
    irreversible_recall_final_phase=("irreversible_recall_final_phase","mean"),
    false_alarm_rate_final_phase=("false_alarm_rate_final_phase","mean"),
    irreversible_core_violation_rate_final_phase=("irreversible_core_violation_rate_final_phase","mean"),
    net_vigilance_value_final_phase=("net_vigilance_value_final_phase","mean"),
    developmental_maturity_final=("developmental_maturity_final","mean"),
    final_TF=("final_TF","mean"),
    final_I=("final_I","mean"),
    final_A=("final_A","mean"),
    final_H=("final_H","mean")
).reset_index()
summary.to_csv(BASE/"summary_v273.csv",index=False)

S=summary.set_index("model")
st="static_quality"; inter="interaction_development"; fb="feedback_development"

corr=float(df["final_ethical_signal_quality"].corr(df["irreversible_recall_final_phase"]))

checks={
    "interaction_crosses_0_80": bool(S.loc[inter,"final_ethical_signal_quality"]>=0.80),
    "feedback_crosses_0_80": bool(S.loc[fb,"final_ethical_signal_quality"]>=0.80),
    "feedback_crosses_earlier_than_interaction": bool(S.loc[fb,"phase_crossing_index"]<=S.loc[inter,"phase_crossing_index"]),
    "feedback_recall_at_least_0_90": bool(S.loc[fb,"irreversible_recall_final_phase"]>=0.90),
    "feedback_better_than_static_recall": bool(S.loc[fb,"irreversible_recall_final_phase"]>=S.loc[st,"irreversible_recall_final_phase"]+0.12),
    "feedback_lower_violations_than_static": bool(S.loc[fb,"irreversible_core_violation_rate_final_phase"]<=S.loc[st,"irreversible_core_violation_rate_final_phase"]-0.08),
    "quality_recall_correlation_positive": bool(corr>0.5)
}
(BASE/"acceptance_v273.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nQUALITY-RECALL CORRELATION:",corr)
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
