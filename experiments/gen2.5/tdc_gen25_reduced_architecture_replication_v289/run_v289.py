
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v289.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]; SL=SPEC["simple_learner"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))

for seed in SPEC["seeds"]:
    env_rng=np.random.default_rng(seed*1000+91)
    env=[]
    for t in range(SPEC["episodes"]):
        pi=t//SPEC["phase_length"]; ph=SPEC["phases"][pi]
        critical=bool(env_rng.random()<E["critical_prob"])
        irreversible=bool(env_rng.random()<E["irreversible_given_critical"]) if critical else bool(env_rng.random()<0.12)
        correct_action=1 if critical else 0
        reward_action=correct_action if env_rng.random()<ph["reward_alignment"] else 1-correct_action
        env.append({
            "pi":pi,"critical":critical,"irreversible":irreversible,
            "correct_action":correct_action,"reward_action":reward_action,
            "r_core":env_rng.random(),"r_world":env_rng.random(),"r_irr":env_rng.random(),
            "r_probe":env_rng.random(),"r_review":env_rng.random(),
            "r_valnoise":env_rng.random()
        })

    for model,cfg in SPEC["models"].items():
        TF=SPEC["initial_state"]["TF"]; I=SPEC["initial_state"]["I"]
        Astate=SPEC["initial_state"]["A"]; H=SPEC["initial_state"]["H"]
        Q=SPEC["initial_state"]["Q"]; mem=0.0

        total_traj=0.0
        trap_capture=trap_events=0
        final_stats=None

        for t,e in enumerate(env):
            ph=SPEC["phases"][e["pi"]]
            if t%SPEC["phase_length"]==0:
                pst={"tp":0,"fn":0,"fp":0,"crit":0,"viol":0,"n":0,"traj":0.0}

            q_use=Q

            norm_acc=clip(q_use-0.10*ph["trap"],0.50,0.98)
            core_signal=e["critical"] if e["r_core"]<norm_acc else (not e["critical"])
            world_signal=e["critical"] if e["r_world"]<ph["world_acc"] else (not e["critical"])
            irr_cue=e["irreversible"] if e["r_irr"]<E["irreversibility_cue_acc"] else (not e["irreversible"])

            score=0.52*int(core_signal)+0.32*int(world_signal)+0.16*int(irr_cue)

            if model=="simple_two_signal_learner":
                anti_capture = 0.08
                if e["reward_action"]==1:
                    score-=anti_capture*(1-ph["reward_alignment"])
            else:
                if cfg["M"]:
                    score+=0.10*mem
                if cfg["Aanti"] and e["reward_action"]==1:
                    score-=D["anti_capture_strength"]*(1-ph["reward_alignment"])
                if cfg["T"]:
                    conflict=(e["reward_action"]!=int(core_signal))
                    if conflict:
                        score+=0.06*(2*int(core_signal)-1)*TF

            threshold=0.48 if irr_cue else 0.60

            if irr_cue and abs(score-threshold)<=0.10:
                probe_acc=clip(q_use+E["probe_bonus"],0.55,0.98)
                probe_pos=e["critical"] if e["r_probe"]<probe_acc else (not e["critical"])
                score=clip(score+(0.20 if probe_pos else -0.16),0,1)
                pst["traj"]-=E["probe_cost"]

            action=1 if score>=threshold else 0
            trigger=(action==1)

            if trigger:
                if e["critical"] and e["irreversible"]: pst["tp"]+=1
                elif not e["critical"]: pst["fp"]+=1
            else:
                if e["critical"] and e["irreversible"]: pst["fn"]+=1

            protected=False
            if trigger:
                protected=e["r_review"]<E["review_success"]
                pst["traj"]-=E["review_cost"]
                if not e["critical"]:
                    pst["traj"]-=E["false_alarm_cost"]

            if e["critical"] and e["irreversible"]:
                pst["crit"]+=1
                if not protected:
                    pst["viol"]+=1
                    pst["traj"]-=E["missed_irreversible_penalty"]
                else:
                    pst["traj"]+=1.0
            else:
                pst["traj"]+=1.0

            normative_correct=(action==e["correct_action"])
            external_success=(action==e["reward_action"])
            validated_success=bool(e["critical"] and e["irreversible"] and protected)
            normative_error=bool(e["critical"] and e["irreversible"] and not protected)

            if ph["trap"]>=0.60 and e["reward_action"]!=e["correct_action"]:
                trap_events+=1
                if action==e["reward_action"]:
                    trap_capture+=1

            if model=="simple_two_signal_learner":
                anti_capture_signal=1.0 if ((not external_success) or e["reward_action"]==e["correct_action"]) else 0.0
                target=0.65*int(normative_correct)+0.35*anti_capture_signal
                Q=clip(Q+SL["eta"]*(target-Q),SL["q_min"],SL["q_max"])
            else:
                if cfg["M"]:
                    mem=D["memory_decay"]*mem + D["memory_gain"]*(1 if normative_correct else -1)
                    mem=clip(mem,-1,1)

                noise=(e["r_valnoise"]-0.5)*0.12
                validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                            +0.18*int((not external_success) or e["reward_action"]==e["correct_action"])
                            +0.10*q_use+noise)
                accepted=(validation>=D["validation_threshold"])

                if cfg["P"] and accepted:
                    gain=D["eta_Q"]*(1.15 if (cfg["T"] and normative_correct) else 1.0)
                    target=1.0 if normative_correct else 0.0
                    Q=clip(Q+gain*(target-Q),0.50,0.94)

                if cfg["T"]:
                    if validated_success and accepted:
                        TF=clip(TF+D["eta_TF"]*(1-TF),0.20,0.95)
                        I=clip(I+D["eta_I"]*(1-I),0.20,0.95)
                        Astate=clip(Astate+D["eta_A"]*(1-Astate),0.20,0.95)
                        H=clip(H-D["eta_H"]*H,0.10,0.90)
                    elif normative_error:
                        H=clip(H+0.004*(1-H),0.10,0.90)

            pst["n"]+=1

            if (t+1)%SPEC["phase_length"]==0:
                rec=pst["tp"]/max(1,pst["tp"]+pst["fn"])
                vr=pst["viol"]/max(1,pst["crit"])
                total_traj+=pst["traj"]
                if e["pi"]==len(SPEC["phases"])-1:
                    final_stats=(rec,vr,Q)

        final_rec,final_vr,final_Q=final_stats
        capture=trap_capture/max(1,trap_events)
        trajectory=total_traj/SPEC["episodes"]
        composite=(0.30*final_rec+0.25*(1-final_vr)+0.20*(1-capture)+0.25*trajectory)

        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":final_rec,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "trajectory_value":trajectory,
            "final_Q":final_Q,
            "composite":composite
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v289.csv",index=False)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    final_Q=("final_Q","mean"),
    composite=("composite","mean")
).reset_index()
summary.to_csv(BASE/"summary_v289.csv",index=False)

S=summary.set_index("model")
pt="PTVM"; fu="FULL"; si="simple_two_signal_learner"

checks={
    "ptvm_beats_full_composite": bool(S.loc[pt,"composite"]>=S.loc[fu,"composite"]+0.01),
    "ptvm_beats_simple_composite": bool(S.loc[pt,"composite"]>=S.loc[si,"composite"]+0.02),
    "ptvm_best_or_tied_recall": bool(S.loc[pt,"final_holdout_recall"]>=summary["final_holdout_recall"].max()-0.01),
    "ptvm_lowest_or_tied_violations": bool(S.loc[pt,"final_holdout_violation_rate"]<=summary["final_holdout_violation_rate"].min()+0.01),
    "ptvm_best_trajectory": bool(S.loc[pt,"trajectory_value"]>=summary["trajectory_value"].max()-1e-12),
    "ptvm_capture_not_materially_worse_than_full": bool(S.loc[pt,"trap_capture_rate"]<=S.loc[fu,"trap_capture_rate"]+0.03),
    "reduced_architecture_replication": bool(
        S.loc[pt,"composite"]>=S.loc[fu,"composite"]+0.01
        and S.loc[pt,"final_holdout_recall"]>=S.loc[fu,"final_holdout_recall"]-0.01
        and S.loc[pt,"final_holdout_violation_rate"]<=S.loc[fu,"final_holdout_violation_rate"]+0.01
        and S.loc[pt,"trajectory_value"]>=S.loc[fu,"trajectory_value"]-0.01
        and S.loc[pt,"trap_capture_rate"]<=S.loc[fu,"trap_capture_rate"]+0.03
    )
}

(BASE/"acceptance_v289.json").write_text(json.dumps({
    "checks":checks,
    "ptvm_minus_full_composite":float(S.loc[pt,"composite"]-S.loc[fu,"composite"]),
    "ptvm_minus_simple_composite":float(S.loc[pt,"composite"]-S.loc[si,"composite"])
},indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
