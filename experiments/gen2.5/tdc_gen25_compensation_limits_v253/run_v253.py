
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).parent
SPEC=json.loads((BASE/"preregistered_spec_v253.json").read_text())
rows=[]

def clip(x,a,b): return max(a,min(b,x))
def sign(x): return 1 if x>=0 else -1
def noisy(rng,truth,acc): return truth if rng.random()<acc else -truth

def maturity(v):
    comps=[v["A_TF"],v["I"],v["C_T"],v["R_C"],1-v["H_self"]]
    G=float(np.prod(np.maximum(comps,1e-9))**(1/5))
    J=G*(0.55*v["A_TF"]+0.45*v["I"])
    rel=clip(0.50+0.18*G+0.42*J,0.50,0.94)
    return G,J,rel

def sensitivity(v):
    return clip(0.46+0.55*(0.35*v["A_TF"]+0.20*v["C_T"]+0.15*v["R_C"]+0.30*v["H_self"]),0.46,0.96)

def make_state(dim,val):
    v=dict(SPEC["high_state"])
    if dim=="one_minus_H_self":
        v["H_self"]=1-val
    else:
        v[dim]=val
    return v

conditions=[]
for dim in SPEC["tested_dimensions"]:
    for val in SPEC["weakness_grid"]:
        conditions.append((dim,val))

n=SPEC["episodes_per_condition"]

for seed in SPEC["seeds"]:
    for ci,(dim,val) in enumerate(conditions):
        rng=np.random.default_rng(seed*1000+ci)
        v=make_state(dim,val)
        G,J,rel=maturity(v)
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
            "dimension":dim,
            "tested_value":val,
            "A_TF":v["A_TF"],
            "I":v["I"],
            "C_T":v["C_T"],
            "R_C":v["R_C"],
            "H_self":v["H_self"],
            "geometric_maturity":G,
            "joint_interaction":J,
            "target_reliability":rel,
            "ethical_sensitivity":sens,
            "ethical_sense_reliability_realized":st["flagok"]/max(1,st["flag"]),
            "decision_accuracy":st["ok"]/n,
            "high_stakes_accuracy":st["high_ok"]/max(1,st["high_n"]),
            "long_term_trajectory_value":st["traj"]/n,
            "deep_path_activation_rate":st["deep"]/n
        })

df=pd.DataFrame(rows)
df.to_csv(BASE/"seed_results_v253.csv",index=False)

summary=df.groupby(["dimension","tested_value"]).agg(
    ethical_sense_reliability_realized=("ethical_sense_reliability_realized","mean"),
    decision_accuracy=("decision_accuracy","mean"),
    long_term_trajectory_value=("long_term_trajectory_value","mean"),
    geometric_maturity=("geometric_maturity","mean"),
    deep_path_activation_rate=("deep_path_activation_rate","mean")
).reset_index()
summary.to_csv(BASE/"summary_v253.csv",index=False)

# Determine monotonicity and critical thresholds
crit_rel=SPEC["critical_deficit_definition"]["reliability_threshold"]
dimension_report={}
monotonic_count=0
critical_count=0
moderate_comp_count=0
severe_fail_count=0
critical_geos=[]

for dim in SPEC["tested_dimensions"]:
    sub=summary[summary["dimension"]==dim].sort_values("tested_value", ascending=False)
    rels=sub["ethical_sense_reliability_realized"].tolist()
    vals=sub["tested_value"].tolist()
    monotonic=all(rels[i]>=rels[i+1]-1e-6 for i in range(len(rels)-1))
    monotonic_count += int(monotonic)

    crit=None
    crit_geo=None
    for _,r in sub.sort_values("tested_value",ascending=False).iterrows():
        if r["ethical_sense_reliability_realized"]<crit_rel:
            crit=float(r["tested_value"])
            crit_geo=float(r["geometric_maturity"])
            break
    if crit is not None:
        critical_count+=1
        critical_geos.append(crit_geo)

    rel_050=float(sub[sub["tested_value"]==0.50]["ethical_sense_reliability_realized"].iloc[0])
    rel_020=float(sub[sub["tested_value"]==0.20]["ethical_sense_reliability_realized"].iloc[0])
    moderate_comp_count += int(rel_050>=crit_rel)
    severe_fail_count += int(rel_020<crit_rel)

    dimension_report[dim]={
        "monotonic":monotonic,
        "critical_tested_value":crit,
        "critical_geometric_maturity":crit_geo,
        "reliability_at_0.50":rel_050,
        "reliability_at_0.20":rel_020
    }

# Sensitivity ranking: higher critical tested value = more sensitive
crit_values={k:v["critical_tested_value"] for k,v in dimension_report.items() if v["critical_tested_value"] is not None}
sorted_crit=sorted(crit_values.items(), key=lambda kv: kv[1], reverse=True)

def spearman(a,b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())

rho=spearman(summary["ethical_sense_reliability_realized"],summary["decision_accuracy"])

# A_TF or I among top 2 sensitivity and higher than at least two other dimensions
atf_i_sensitive=False
for dim in ["A_TF","I"]:
    if dim in crit_values:
        lower=sum(1 for od,v in crit_values.items() if od!=dim and crit_values[dim]>v)
        if lower>=2:
            atf_i_sensitive=True

geo_band=bool(len(critical_geos)>0 and all(0.45<=g<=0.70 for g in critical_geos))

checks={
    "monotonic_reliability_decline": bool(monotonic_count>=4),
    "critical_threshold_exists": bool(critical_count>=3),
    "ATF_or_I_most_sensitive": bool(atf_i_sensitive),
    "high_other_capacities_compensate_moderate_weakness": bool(moderate_comp_count>=3),
    "severe_weakness_not_fully_compensable": bool(severe_fail_count>=4),
    "geometric_transition_band": bool(geo_band),
    "accuracy_tracks_reliability": bool(rho>=0.85)
}

(BASE/"dimension_report_v253.json").write_text(json.dumps(dimension_report,indent=2))
(BASE/"acceptance_v253.json").write_text(json.dumps(checks,indent=2))

print(summary.to_string(index=False))
print("\nDIMENSION REPORT")
print(json.dumps(dimension_report,indent=2))
print("\nSpearman reliability vs accuracy:",rho)
print("\nACCEPTANCE")
for k,v in checks.items():
    print(f"{k}: {v}")
print(f"passed={sum(checks.values())}/{len(checks)}")
