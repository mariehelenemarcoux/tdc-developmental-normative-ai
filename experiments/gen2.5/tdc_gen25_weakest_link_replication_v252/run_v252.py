
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SPEC = json.loads((BASE/"preregistered_spec_v252.json").read_text())
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

def maturity(v):
    comps=[v["A_TF"],v["I"],v["C_T"],v["R_C"],1-v["H_self"]]
    weakest=min(comps)
    mean=float(np.mean(comps))
    geo=float(np.prod(np.maximum(comps,1e-9))**(1/5))
    joint=geo*(0.55*v["A_TF"]+0.45*v["I"])
    return weakest,mean,geo,joint

def reliability(geo,joint):
    return clip(0.50+0.18*geo+0.42*joint,0.50,0.94)

def sensitivity(v):
    return clip(0.46+0.55*(0.35*v["A_TF"]+0.20*v["C_T"]+0.15*v["R_C"]+0.30*v["H_self"]),0.46,0.96)

profile_id=0
for seed in SPEC["seeds"]:
    rng_profile=np.random.default_rng(seed)
    for j in range(SPEC["profiles_per_seed"]):
        profile_id += 1
        v={
            "A_TF": rng_profile.uniform(0.20,0.95),
            "I": rng_profile.uniform(0.20,0.95),
            "C_T": rng_profile.uniform(0.20,0.95),
            "R_C": rng_profile.uniform(0.20,0.95),
            "H_self": rng_profile.uniform(0.05,0.80)
        }
        weakest,mean,geo,joint=maturity(v)
        rel=reliability(geo,joint)
        sens=sensitivity(v)

        rng=np.random.default_rng(seed*1000+j)
        n=SPEC["episodes_per_profile"]
        st={"ok":0,"flag":0,"flagok":0,"high_n":0,"high_ok":0,"traj":0.0,"deep":0}

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
            "profile_id":profile_id,
            "seed":seed,
            **v,
            "weakest_link":weakest,
            "mean_maturity":mean,
            "geometric_maturity":geo,
            "joint_interaction":joint,
            "target_reliability":rel,
            "ethical_sensitivity":sens,
            "ethical_sense_reliability_realized":st["flagok"]/max(1,st["flag"]),
            "decision_accuracy":st["ok"]/n,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/n,
            "deep_path_activation_rate":st["deep"]/n
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"profile_results_v252.csv",index=False)

# Spearman via rank-Pearson
def spearman(a,b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())

rho_weak=spearman(df["weakest_link"],df["ethical_sense_reliability_realized"])
rho_mean=spearman(df["mean_maturity"],df["ethical_sense_reliability_realized"])
rho_geo=spearman(df["geometric_maturity"],df["ethical_sense_reliability_realized"])
rho_acc=spearman(df["ethical_sense_reliability_realized"],df["decision_accuracy"])

df["weak_decile"]=pd.qcut(df["weakest_link"],10,labels=False,duplicates="drop")
bottom=df[df["weak_decile"]==df["weak_decile"].min()]["ethical_sense_reliability_realized"].mean()
top=df[df["weak_decile"]==df["weak_decile"].max()]["ethical_sense_reliability_realized"].mean()
decile_gap=top-bottom

high_mean_low_min=df[(df["mean_maturity"]>=0.65)&(df["weakest_link"]<=0.35)]
high_min=df[df["weakest_link"]>=0.60]

summary={
    "n_profiles":len(df),
    "rho_weakest_reliability":rho_weak,
    "rho_mean_reliability":rho_mean,
    "rho_geometric_reliability":rho_geo,
    "rho_reliability_accuracy":rho_acc,
    "bottom_weakest_decile_reliability":bottom,
    "top_weakest_decile_reliability":top,
    "weakest_decile_gap":decile_gap,
    "high_mean_low_min_n":len(high_mean_low_min),
    "high_mean_low_min_reliability":high_mean_low_min["ethical_sense_reliability_realized"].mean() if len(high_mean_low_min) else None,
    "high_min_n":len(high_min),
    "high_min_reliability":high_min["ethical_sense_reliability_realized"].mean() if len(high_min) else None
}
(BASE/"summary_v252.json").write_text(json.dumps(summary,indent=2))

checks={
    "weakest_link_correlates_strongly": bool(rho_weak>=0.70),
    "weakest_link_beats_mean": bool(rho_weak>=rho_mean+0.08),
    "geometric_maturity_correlates_strongly": bool(rho_geo>=0.80),
    "bottom_weakest_decile_penalty": bool(decile_gap>=0.12),
    "high_mean_low_min_is_unreliable": bool(len(high_mean_low_min)>0 and high_mean_low_min["ethical_sense_reliability_realized"].mean()<=0.72),
    "high_min_profiles_are_reliable": bool(len(high_min)>0 and high_min["ethical_sense_reliability_realized"].mean()>=0.80),
    "accuracy_tracks_reliability": bool(rho_acc>=0.75)
}
(BASE/"acceptance_v252.json").write_text(json.dumps(checks,indent=2))

agg=df[[
    "weakest_link","mean_maturity","geometric_maturity","joint_interaction",
    "ethical_sense_reliability_realized","decision_accuracy","long_term_trajectory_value"
]].describe().T
agg.to_csv(BASE/"descriptive_summary_v252.csv")

print(json.dumps(summary,indent=2))
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
