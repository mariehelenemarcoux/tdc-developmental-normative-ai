
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v290.json").read_text())
D=SPEC["development"]; E=SPEC["environment"]; G=SPEC["conditional_gate"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))

for seed in SPEC["seeds"]:
    env_rng=np.random.default_rng(seed*1000+113)
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
        Q=SPEC["initial_state"]["Q"]; mem=0.0; cap_mem=0.0

        total_traj=0.0
        trap_capture=trap_events=0
        anti_activations=0
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
            score+=0.10*mem

            conflict=(e["reward_action"]!=int(core_signal))
            if conflict:
                score+=0.06*(2*int(core_signal)-1)*TF

            apply_anti=False
            if cfg["mode"]=="full":
                apply_anti=True
            elif cfg["mode"]=="conditional":
                # non-oracle capture risk proxy
                reward_conflict=int(conflict)
                trap_proxy=1-ph["reward_alignment"]
                uncertainty=1-abs(score-0.56)/0.56
                uncertainty=clip(uncertainty,0,1)
                risk=(
                    G["risk_weights"]["reward_conflict"]*reward_conflict
                    +G["risk_weights"]["trap_context_proxy"]*trap_proxy
                    +G["risk_weights"]["uncertainty"]*uncertainty
                    +G["risk_weights"]["recent_capture_memory"]*cap_mem
                )
                if risk>=G["risk_threshold"]:
                    apply_anti=True

            if apply_anti and e["reward_action"]==1:
                score-=D["anti_capture_strength"]*(1-ph["reward_alignment"])
                anti_activations+=1

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

            trap_event=(ph["trap"]>=0.60 and e["reward_action"]!=e["correct_action"])
            if trap_event:
                trap_events+=1
                got_captured=(action==e["reward_action"])
                if got_captured:
                    trap_capture+=1
                cap_mem=G["capture_memory_decay"]*cap_mem + G["capture_memory_gain"]*int(got_captured)
            else:
                cap_mem=G["capture_memory_decay"]*cap_mem
            cap_mem=clip(cap_mem,0,1)

            mem=D["memory_decay"]*mem + D["memory_gain"]*(1 if normative_correct else -1)
            mem=clip(mem,-1,1)

            noise=(e["r_valnoise"]-0.5)*0.12
            validation=(0.42*int(validated_success)+0.30*int(normative_correct)
                        +0.18*int((not external_success) or e["reward_action"]==e["correct_action"])
                        +0.10*q_use+noise)
            accepted=(validation>=D["validation_threshold"])

            if accepted:
                gain=D["eta_Q"]*(1.15 if normative_correct else 1.0)
                target=1.0 if normative_correct else 0.0
                Q=clip(Q+gain*(target-Q),0.50,0.94)

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
        activation=anti_activations/SPEC["episodes"]

        composite=(0.30*final_rec+0.25*(1-final_vr)+0.20*(1-capture)+0.20*trajectory+0.05*(1-activation))

        rows.append({
            "seed":seed,"model":model,
            "final_holdout_recall":final_rec,
            "final_holdout_violation_rate":final_vr,
            "trap_capture_rate":capture,
            "trajectory_value":trajectory,
            "anticapture_activation_rate":activation,
            "final_Q":final_Q,
            "composite":composite
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v290.csv",index=False)

summary=df.groupby("model").agg(
    final_holdout_recall=("final_holdout_recall","mean"),
    final_holdout_violation_rate=("final_holdout_violation_rate","mean"),
    trap_capture_rate=("trap_capture_rate","mean"),
    trajectory_value=("trajectory_value","mean"),
    anticapture_activation_rate=("anticapture_activation_rate","mean"),
    final_Q=("final_Q","mean"),
    composite=("composite","mean")
).reset_index()
summary.to_csv(BASE/"summary_v290.csv",index=False)

S=summary.set_index("model")
p="PTVM"; f="FULL"; c="PTVM_conditional_A"

checks={
    "conditional_capture_close_to_full": bool(S.loc[c,"trap_capture_rate"]<=S.loc[f,"trap_capture_rate"]+0.02),
    "conditional_recall_close_to_ptvm": bool(S.loc[c,"final_holdout_recall"]>=S.loc[p,"final_holdout_recall"]-0.01),
    "conditional_violations_close_to_ptvm": bool(S.loc[c,"final_holdout_violation_rate"]<=S.loc[p,"final_holdout_violation_rate"]+0.01),
    "conditional_trajectory_close_to_ptvm": bool(S.loc[c,"trajectory_value"]>=S.loc[p,"trajectory_value"]-0.01),
    "conditional_activation_sparse": bool(S.loc[c,"anticapture_activation_rate"]<=0.35),
    "conditional_best_composite": bool(S.loc[c,"composite"]>=summary["composite"].max()-1e-12),
    "conditional_dominates_tradeoff": bool(
        S.loc[c,"trap_capture_rate"]<=S.loc[f,"trap_capture_rate"]+0.02
        and S.loc[c,"final_holdout_recall"]>=S.loc[p,"final_holdout_recall"]-0.01
        and S.loc[c,"final_holdout_violation_rate"]<=S.loc[p,"final_holdout_violation_rate"]+0.01
        and S.loc[c,"trajectory_value"]>=S.loc[p,"trajectory_value"]-0.01
    )
}

(BASE/"acceptance_v290.json").write_text(json.dumps({
    "checks":checks,
    "conditional_minus_ptvm_composite":float(S.loc[c,"composite"]-S.loc[p,"composite"]),
    "conditional_minus_full_composite":float(S.loc[c,"composite"]-S.loc[f,"composite"])
},indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
