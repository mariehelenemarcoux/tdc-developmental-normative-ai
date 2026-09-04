
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v264.json").read_text())
E=SPEC["environment"]
RFL=SPEC["reflection"]
LR=SPEC["learning_rates"]
INIT=SPEC["initial_state"]
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

for seed in SPEC["seeds"]:
    for mi,model in enumerate(SPEC["models"].keys()):
        rng=np.random.default_rng(seed*100+mi)
        W=INIT["world_model_skill"]
        R=INIT["ethical_reliability"]
        I=INIT["integration"]
        H=INIT["H_self"]
        T=INIT["transfer_memory"]

        total_ok=0
        p2_ok=0
        p2_n=0
        switched_ok=0
        switched_n=0
        repeat_errors=0
        error_context={}
        refl_steps=0
        qualifying_errors=0
        traj=0.0

        for t in range(SPEC["episodes"]):
            phase=1 if t<SPEC["phase1_episodes"] else 2

            mech_switch = bool(rng.random()<E["latent_mechanism_switch_probability"])
            context=int(rng.integers(0,8))
            fact=1 if rng.random()<0.5 else -1
            norm=1 if rng.random()<0.5 else -1

            # In switched mechanism, interaction flips causal importance.
            if mech_switch:
                y=sign(0.25*fact+0.95*norm-0.55*fact*norm)
            else:
                y=sign(0.65*fact+0.75*norm+0.30*fact*norm)

            fact_obs=noisy(rng,fact,E["world_signal_accuracy"])
            norm_obs=noisy(rng,norm,E["normative_signal_accuracy"])
            world_est=fact_obs if rng.random()<W else -fact_obs
            ethical_est=norm_obs if rng.random()<R else -norm_obs

            # Transfer memory helps switched mechanisms specifically.
            transfer_weight = 0.15 + 0.55*T if mech_switch else 0.10*T
            score=0.52*world_est+0.62*ethical_est
            if mech_switch:
                score += transfer_weight * (-world_est*ethical_est)
            else:
                score += transfer_weight * (world_est*ethical_est)
            action=sign(score)

            ok=(action==y)
            total_ok+=int(ok)
            if phase==2:
                p2_n+=1
                p2_ok+=int(ok)
            if mech_switch and phase==2:
                switched_n+=1
                switched_ok+=int(ok)

            if ok:
                traj+=1.0
            else:
                high=bool(rng.random()<E["high_stakes_probability"])
                if high:
                    traj-=E["severe_error_penalty"]

            # repeated error tracking by context/mechanism signature
            key=(context,mech_switch)
            if not ok and error_context.get(key,0)>0:
                repeat_errors+=1
            if not ok:
                error_context[key]=error_context.get(key,0)+1

            # post-action audit
            if not ok:
                surprise=clip(0.45+0.35*rng.random()+0.20*(1-W),0,1)
                high=bool(rng.random()<E["high_stakes_probability"])
                repeated=error_context.get(key,0)>=2
                qualifies = high or surprise>0.45 or repeated

                if qualifies:
                    qualifying_errors+=1

                    # base correction
                    W=clip(W+LR["base_world_update"]*(1-W),0.50,0.95)
                    R=clip(R+LR["base_ethical_update"]*(1-R),0.50,0.95)

                    if model=="no_reflection":
                        cycles=0

                    elif model=="fixed_reflection_x6":
                        cycles=RFL["fixed_cycles"]
                        for c in range(cycles):
                            novelty=clip(0.28+0.10*(1-T)+0.05*rng.normal(),0,1)
                            W=clip(W+LR["structured_reflection_gain"]*novelty*(1-W),0.50,0.95)
                            R=clip(R+0.7*LR["structured_reflection_gain"]*novelty*(1-R),0.50,0.95)
                            T=clip(T+LR["transfer_gain"]*novelty*(1-T),0,0.95)
                            I=clip(I+0.003*novelty,0.20,0.95)
                            H=clip(H-LR["H_down"]*novelty,0.10,0.90)

                    elif model=="adaptive_reflection":
                        cycles=0
                        low_gain_streak=0
                        last_gain=1.0
                        for c in range(RFL["adaptive_max_cycles"]):
                            # novelty decays as lesson becomes exhausted
                            novelty=clip((0.36*np.exp(-0.38*c)) + 0.13*(1-T) + 0.04*rng.normal(),0,1)
                            causal_clarity=clip(0.30+0.22*W+0.12*rng.normal(),0,1)
                            transferability=clip(0.20+0.38*(1-T)+0.08*rng.normal(),0,1)
                            gain=(novelty+causal_clarity+transferability)/3.0
                            cycles+=1

                            W=clip(W+LR["structured_reflection_gain"]*gain*(1-W),0.50,0.95)
                            R=clip(R+0.7*LR["structured_reflection_gain"]*gain*(1-R),0.50,0.95)
                            T=clip(T+LR["transfer_gain"]*gain*(1-T),0,0.95)
                            I=clip(I+0.0035*gain,0.20,0.95)
                            H=clip(H-LR["H_down"]*gain,0.10,0.90)

                            marginal=abs(gain-last_gain)
                            if gain<RFL["adaptive_epsilon"] or marginal<RFL["adaptive_epsilon"]:
                                low_gain_streak+=1
                            else:
                                low_gain_streak=0
                            last_gain=gain
                            if low_gain_streak>=RFL["adaptive_patience"]:
                                break

                    else: # brute_rumination_x8
                        cycles=8
                        for c in range(cycles):
                            # repeated replay gives tiny early update but adds noise and H_self
                            replay_gain=max(0.0,0.08-0.01*c)
                            W=clip(W+0.35*LR["structured_reflection_gain"]*replay_gain*(1-W)
                                   +rng.normal(0,LR["rumination_noise"]),0.50,0.95)
                            R=clip(R+0.25*LR["structured_reflection_gain"]*replay_gain*(1-R)
                                   +rng.normal(0,LR["rumination_noise"]),0.50,0.95)
                            H=clip(H+LR["H_up_rumination"]*(0.65+0.05*c),0.10,0.95)

                    refl_steps+=cycles
                    traj-=cycles*E["reflection_step_cost"]

        rows.append({
            "seed":seed,
            "model":model,
            "phase2_accuracy":p2_ok/max(1,p2_n),
            "repeat_error_rate":repeat_errors/max(1,SPEC["episodes"]),
            "transfer_accuracy_on_switched_mechanisms":switched_ok/max(1,switched_n),
            "reflection_steps_per_error":refl_steps/max(1,qualifying_errors),
            "long_term_trajectory_value":traj/SPEC["episodes"],
            "final_H_self":H,
            "final_transfer_memory":T,
            "final_world_model_skill":W,
            "final_ethical_reliability":R
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v264.csv",index=False)

summary=df.groupby("model").agg(
    phase2_accuracy=("phase2_accuracy","mean"),
    repeat_error_rate=("repeat_error_rate","mean"),
    transfer_accuracy_on_switched_mechanisms=("transfer_accuracy_on_switched_mechanisms","mean"),
    reflection_steps_per_error=("reflection_steps_per_error","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    final_H_self=("final_H_self","mean"),
    final_transfer_memory=("final_transfer_memory","mean"),
    final_world_model_skill=("final_world_model_skill","mean"),
    final_ethical_reliability=("final_ethical_reliability","mean")
).reset_index()
summary.to_csv(BASE/"summary_v264.csv",index=False)

S=summary.set_index("model")
a="adaptive_reflection"; n="no_reflection"; f="fixed_reflection_x6"; b="brute_rumination_x8"
best_nf=max(S.loc[n,"long_term_trajectory_value"],S.loc[f,"long_term_trajectory_value"])

checks={
    "adaptive_beats_none_transfer": bool(S.loc[a,"transfer_accuracy_on_switched_mechanisms"]>=S.loc[n,"transfer_accuracy_on_switched_mechanisms"]+0.07),
    "adaptive_beats_fixed_transfer": bool(S.loc[a,"transfer_accuracy_on_switched_mechanisms"]>=S.loc[f,"transfer_accuracy_on_switched_mechanisms"]+0.02),
    "adaptive_reduces_repeat_errors": bool(S.loc[a,"repeat_error_rate"]<=S.loc[n,"repeat_error_rate"]-0.06),
    "adaptive_more_efficient_than_fixed": bool(S.loc[a,"reflection_steps_per_error"]<=S.loc[f,"reflection_steps_per_error"]-1.0),
    "adaptive_longterm_not_worse": bool(S.loc[a,"long_term_trajectory_value"]>=best_nf-0.01),
    "brute_rumination_not_best": bool(S.loc[b,"transfer_accuracy_on_switched_mechanisms"]<S.loc[a,"transfer_accuracy_on_switched_mechanisms"]),
    "brute_rumination_increases_Hself": bool(S.loc[b,"final_H_self"]>=S.loc[a,"final_H_self"]+0.08)
}
(BASE/"acceptance_v264.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
