
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v251.json").read_text())
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

def multiplicative_parts(v):
    vals=[max(v["A_TF"],1e-9),max(v["I"],1e-9),max(v["C_T"],1e-9),max(v["R_C"],1e-9),max(1-v["H_self"],1e-9)]
    P=np.prod(vals)**(1/5)
    Q=P*(0.55*v["A_TF"]+0.45*v["I"])
    rel=clip(0.50+0.18*P+0.42*Q,0.50,0.94)
    return P,Q,rel

def sensitivity(v):
    return clip(0.46+0.55*(0.35*v["A_TF"]+0.20*v["C_T"]+0.15*v["R_C"]+0.30*v["H_self"]),0.46,0.96)

base_state=SPEC["base_D5_state"]
weak=SPEC["weak_intermediate_state_for_superpower_test"]

def condition_state(name):
    v=dict(base_state)
    if name=="ablate_A_TF": v["A_TF"]=0.35
    elif name=="ablate_I": v["I"]=0.35
    elif name=="ablate_C_T": v["C_T"]=0.35
    elif name=="ablate_R_C": v["R_C"]=0.35
    elif name=="ablate_low_Hself": v["H_self"]=0.65
    elif name=="single_superpower_A_TF":
        v={"A_TF":0.95,"I":weak["I"],"C_T":weak["C_T"],"R_C":weak["R_C"],"H_self":weak["H_self"]}
    return v

conditions=list(SPEC["conditions"].keys())
n=SPEC["episodes_per_condition"]

for seed in SPEC["seeds"]:
    for ci,cond in enumerate(conditions):
        rng=np.random.default_rng(seed*100+ci)
        v=condition_state(cond)
        P,Q,rel=multiplicative_parts(v)
        sens=sensitivity(v)
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
            "condition":cond,
            "A_TF":v["A_TF"],
            "I":v["I"],
            "C_T":v["C_T"],
            "R_C":v["R_C"],
            "H_self":v["H_self"],
            "joint_maturity_product":P,
            "joint_interaction":Q,
            "target_reliability":rel,
            "ethical_sensitivity":sens,
            "decision_accuracy":st["ok"]/n,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/n,
            "ethical_sense_reliability_realized":st["flagok"]/max(1,st["flag"]),
            "deep_path_activation_rate":st["deep"]/n
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v251.csv",index=False)

summary=df.groupby("condition").agg(
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
summary.to_csv(BASE/"summary_v251.csv",index=False)

S=summary.set_index("condition")
full=S.loc["full_D5"]
single_ablations=["ablate_A_TF","ablate_I","ablate_C_T","ablate_R_C","ablate_low_Hself"]

reliability_drops={c: full["ethical_sense_reliability_realized"]-S.loc[c,"ethical_sense_reliability_realized"] for c in single_ablations}
accuracy_drops={c: full["decision_accuracy"]-S.loc[c,"decision_accuracy"] for c in single_ablations}
traj_drops={c: full["long_term_trajectory_value"]-S.loc[c,"long_term_trajectory_value"] for c in single_ablations}

# Spearman across all non-oracle conditions
ranked=summary[["joint_maturity_product","ethical_sense_reliability_realized"]].rank()
spearman=ranked.corr(method="pearson").iloc[0,1]

checks={
    "full_D5_accuracy_above_075": bool(full["decision_accuracy"]>=0.75),
    "each_single_ablation_hurts_reliability": bool(all(v>=0.05 for v in reliability_drops.values())),
    "at_least_three_ablations_hurt_accuracy": bool(sum(v>=0.04 for v in accuracy_drops.values())>=3),
    "integration_is_critical": bool(traj_drops["ablate_I"]>=0.06),
    "hself_is_critical": bool(traj_drops["ablate_low_Hself"]>=0.06),
    "single_superpower_not_enough": bool(S.loc["single_superpower_A_TF","decision_accuracy"]<=full["decision_accuracy"]-0.08),
    "joint_product_tracks_reliability": bool(spearman>=0.80)
}
(BASE/"acceptance_v251.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nRELIABILITY DROPS VS FULL D5")
for k,v in reliability_drops.items(): print(k, v)
print("\nACCURACY DROPS VS FULL D5")
for k,v in accuracy_drops.items(): print(k, v)
print("\nTRAJECTORY DROPS VS FULL D5")
for k,v in traj_drops.items(): print(k, v)
print("\nSpearman product vs reliability:",spearman)
print("\nACCEPTANCE")
for k,v in checks.items(): print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
