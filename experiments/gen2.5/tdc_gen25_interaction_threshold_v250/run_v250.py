
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v250.json").read_text())
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

initial=SPEC["developmental_dynamics"]["initial_means"]
updates=SPEC["developmental_dynamics"]["stage_updates"]
noise_sd=SPEC["developmental_dynamics"]["noise_sd"]
stages=SPEC["stages"]
n=SPEC["episodes_per_stage"]

def state_for(rng,stage):
    vals={}
    for k in initial:
        vals[k]=initial[k]+sum(updates[k][:stage])+rng.normal(0,noise_sd)
    for k in ["A_TF","I","C_T","R_C","H_self"]:
        vals[k]=clip(vals[k],0,1)
    return vals

def linear_rel(v):
    z=0.27*v["A_TF"]+0.28*v["I"]+0.18*v["C_T"]+0.17*v["R_C"]+0.10*(1-v["H_self"])
    return clip(0.50+0.43*z,0.50,0.94)

def multiplicative_parts(v):
    vals=[max(v["A_TF"],1e-9),max(v["I"],1e-9),max(v["C_T"],1e-9),max(v["R_C"],1e-9),max(1-v["H_self"],1e-9)]
    P=np.prod(vals)**(1/5)
    Q=P*(0.55*v["A_TF"]+0.45*v["I"])
    rel=clip(0.50+0.18*P+0.42*Q,0.50,0.94)
    return P,Q,rel

def sensitivity(v):
    return clip(0.46+0.55*(0.35*v["A_TF"]+0.20*v["C_T"]+0.15*v["R_C"]+0.30*v["H_self"]),0.46,0.96)

models=["static_reliability","linear_internal_map","multiplicative_threshold"]

for seed in SPEC["seeds"]:
    state_rng=np.random.default_rng(seed)
    for stage in stages:
        v=state_for(state_rng,stage)
        P,Q,mrel=multiplicative_parts(v)
        rels={
            "static_reliability":0.62,
            "linear_internal_map":linear_rel(v),
            "multiplicative_threshold":mrel
        }
        sens=sensitivity(v)

        for mi,model in enumerate(models):
            rng=np.random.default_rng(seed*1000+stage*10+mi)
            rel=rels[model]
            st={"ok":0,"high_n":0,"high_ok":0,"traj":0.0,"flag":0,"flagok":0,"deep":0}
            for _ in range(n):
                w=1 if rng.random()<0.5 else -1
                c=1 if rng.random()<0.5 else -1
                l=1 if rng.random()<0.5 else -1
                s=1 if rng.random()<0.5 else -1
                k=1 if rng.random()<0.5 else -1
                latent_score=0.40*w+0.35*c+0.65*l+0.55*s+0.72*k+0.45*l*k+0.30*s*k
                y=sign(latent_score)
                obs=[
                    noisy(rng,w,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,c,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,l,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,s,SPEC["environment"]["base_signal_accuracy"]),
                    noisy(rng,k,SPEC["environment"]["base_signal_accuracy"])
                ]
                majority=sign(sum(obs))
                conflict=len(set(obs))>1
                high=bool(rng.random()<SPEC["environment"]["high_stakes_probability"])
                flag=bool(conflict and rng.random()<sens)
                action=majority
                deep=False
                if flag:
                    st["flag"]+=1
                    interpreted=y if rng.random()<rel else -y
                    st["flagok"]+=int(interpreted==y)
                    authority=clip(
                        0.30*v["A_TF"]+0.30*v["I"]+0.20*(1-v["H_self"])+0.10*v["C_T"]+0.10*(1.0 if high else 0.0),
                        0,1
                    )
                    deep=authority>=0.45
                    if deep:
                        action=interpreted
                ok=(action==y)
                st["ok"]+=int(ok)
                if high:
                    st["high_n"]+=1
                    st["high_ok"]+=int(ok)
                if ok:
                    st["traj"]+=SPEC["environment"]["correct_reward"]
                elif high:
                    st["traj"]-=SPEC["environment"]["severe_error_penalty"]
                if deep:
                    st["deep"]+=1
                    st["traj"]-=SPEC["environment"]["deep_cost"]
            rows.append({
                "seed":seed,
                "model":model,
                "stage":stage,
                "A_TF":v["A_TF"],"I":v["I"],"C_T":v["C_T"],"R_C":v["R_C"],"H_self":v["H_self"],
                "joint_maturity_product":P,
                "joint_interaction":Q,
                "ethical_sensitivity":sens,
                "target_reliability":rel,
                "decision_accuracy":st["ok"]/n,
                "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
                "long_term_trajectory_value":st["traj"]/n,
                "ethical_sense_reliability_realized":st["flagok"]/max(1,st["flag"]),
                "deep_path_activation_rate":st["deep"]/n
            })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v250.csv",index=False)

summary=df.groupby(["model","stage"]).agg(
    decision_accuracy=("decision_accuracy","mean"),
    decision_accuracy_std=("decision_accuracy","std"),
    high_stakes_accuracy=("high_stakes_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    ethical_sense_reliability_realized=("ethical_sense_reliability_realized","mean"),
    ethical_sensitivity=("ethical_sensitivity","mean"),
    joint_maturity_product=("joint_maturity_product","mean"),
    joint_interaction=("joint_interaction","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean")
).reset_index()
summary.to_csv(BASE/"summary_v250.csv",index=False)

S=summary.set_index(["model","stage"])
m2=S.loc[("multiplicative_threshold",2)]
m3=S.loc[("multiplicative_threshold",3)]
m4=S.loc[("multiplicative_threshold",4)]
m5=S.loc[("multiplicative_threshold",5)]
l2=S.loc[("linear_internal_map",2)]
l5=S.loc[("linear_internal_map",5)]

checks={
    "multiplicative_final_accuracy_above_078": bool(m5["decision_accuracy"]>=0.78),
    "multiplicative_late_jump": bool(m5["decision_accuracy"]-m4["decision_accuracy"]>=0.10),
    "multiplicative_reliability_jump": bool(m5["ethical_sense_reliability_realized"]-m4["ethical_sense_reliability_realized"]>=0.10),
    "multiplicative_beats_linear_stage5": bool(m5["long_term_trajectory_value"]>=l5["long_term_trajectory_value"]+0.02),
    "multiplicative_not_better_early": bool(m2["decision_accuracy"]<=l2["decision_accuracy"]+0.02),
    "sensitivity_precedes_reliability": bool(m3["ethical_sensitivity"]>=0.70 and m3["ethical_sense_reliability_realized"]<=0.70),
    "joint_product_rises_nontrivially": bool(m5["joint_maturity_product"]-m4["joint_maturity_product"]>=0.10)
}
(BASE/"acceptance_v250.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
print("\nKEY SHAPE")
print("D4->D5 accuracy gain:", m5["decision_accuracy"]-m4["decision_accuracy"])
print("D4->D5 reliability gain:", m5["ethical_sense_reliability_realized"]-m4["ethical_sense_reliability_realized"])
print("D4->D5 joint product gain:", m5["joint_maturity_product"]-m4["joint_maturity_product"])
print("Stage5 multiplicative vs linear trajectory gap:", m5["long_term_trajectory_value"]-l5["long_term_trajectory_value"])
