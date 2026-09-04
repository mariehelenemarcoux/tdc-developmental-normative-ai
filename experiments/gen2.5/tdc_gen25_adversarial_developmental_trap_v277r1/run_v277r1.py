import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
BASE=Path(__file__).parent
SPEC=json.loads((BASE/'preregistered_spec_v277r1.json').read_text())
phase_reward=np.array([p['reward_alignment'] for p in SPEC['phases']],dtype=np.float64)
phase_world=np.array([p['world_acc'] for p in SPEC['phases']],dtype=np.float64)
phase_false=np.array([p['false_progress'] for p in SPEC['phases']],dtype=np.float64)
phase_trap=np.array([p['trap_strength'] for p in SPEC['phases']],dtype=np.float64)

@njit
def clip(x,a,b):
    return a if x<a else (b if x>b else x)
@njit
def noisy(truth,acc):
    return truth if np.random.random()<acc else (not truth)

@njit
def run_one(seed, mi, episodes, phase_length, reward_arr, world_arr, false_arr, trap_arr):
    np.random.seed(seed*100+mi)
    TF=0.24; I=0.34; A=0.40; H=0.62; Q=0.58; success_Q=Q
    total_traj=0.0; trap_capture=0; trap_events=0; false_consol=0; consol_attempts=0; provisional_debt=0.0
    phase_rec=np.zeros(6); phase_far=np.zeros(6); phase_vr=np.zeros(6); phase_cap=np.zeros(6); phase_traj=np.zeros(6); phase_q=np.zeros(6)
    tp=fn=fp=crit=viol=n=cap=tev=0; ptraj=0.0
    for t in range(episodes):
        pi=t//phase_length
        if t%phase_length==0:
            tp=fn=fp=crit=viol=n=cap=tev=0; ptraj=0.0
        reward_alignment=reward_arr[pi]; world_acc=world_arr[pi]; false_progress=false_arr[pi]; trap_strength=trap_arr[pi]
        in_trap=trap_strength>=0.60
        if mi==0:
            frac=t/(episodes-1.0); q_use=0.58+(0.92-0.58)*frac
        elif mi==1:
            q_use=success_Q
        else:
            mprod=(TF*I*A*(1.0-H))**0.25
            sig=1.0/(1.0+math.exp(-8.0*(mprod-0.58)))
            Q=clip(0.52+(0.94-0.52)*sig,0.52,0.94); q_use=Q
        critical=np.random.random()<0.22
        irreversible=(np.random.random()<0.48) if critical else (np.random.random()<0.12)
        correct_action=1 if critical else 0
        reward_action=correct_action if np.random.random()<reward_alignment else 1-correct_action
        norm_acc=clip(q_use-0.10*trap_strength,0.50,0.98)
        core_signal=noisy(critical,norm_acc); world_signal=noisy(critical,world_acc); irrev_cue=noisy(irreversible,0.74)
        score=0.52*(1 if core_signal else 0)+0.32*(1 if world_signal else 0)+0.16*(1 if irrev_cue else 0)
        threshold=0.48 if irrev_cue else 0.60
        if irrev_cue and abs(score-threshold)<=0.10:
            probe_acc=clip(q_use+0.10,0.55,0.98); probe_pos=noisy(critical,probe_acc)
            score=clip(score+(0.20 if probe_pos else -0.16),0.0,1.0); ptraj-=0.028
        action=1 if score>=threshold else 0; trigger=action==1
        if trigger:
            if critical and irreversible: tp+=1
            elif not critical: fp+=1
        else:
            if critical and irreversible: fn+=1
        protected=False
        if trigger:
            protected=np.random.random()<0.84; ptraj-=0.016
            if not critical: ptraj-=0.22
        if critical and irreversible:
            crit+=1
            if not protected: viol+=1; ptraj-=5.4
            else: ptraj+=1.0
        else: ptraj+=1.0
        external_success=(action==reward_action); validated_success=(critical and irreversible and protected); normative_correct=(action==correct_action); normative_error=(critical and irreversible and not protected)
        if in_trap and reward_action!=correct_action:
            trap_events+=1; tev+=1
            if action==reward_action: trap_capture+=1; cap+=1
        if mi==1:
            if external_success: success_Q=clip(success_Q+0.007*(1.0-success_Q),0.50,0.94)
            else: success_Q=clip(success_Q-0.002*success_Q,0.50,0.94)
        elif mi>=2:
            candidate_signal=0.40*(1 if validated_success else 0)+0.30*(1 if external_success else 0)+0.20*(1.0-H)+0.10*q_use
            if np.random.random()<false_progress: candidate_signal+=0.28
            accepted=True; validation_score=1.0
            if mi!=2:
                validation_score=0.40*(1 if validated_success else 0)+0.30*(1 if normative_correct else 0)+0.20*(1 if ((not external_success) or reward_action==correct_action) else 0)+0.10*q_use+np.random.normal(0.0,0.06)
                accepted=validation_score>=0.70
            if candidate_signal>0.35:
                consol_attempts+=1
                if accepted:
                    if normative_correct:
                        TF=clip(TF+0.006*(1-TF),0.20,0.95); I=clip(I+0.007*(1-I),0.20,0.95); A=clip(A+0.006*(1-A),0.20,0.95); H=clip(H-0.006*H,0.10,0.90)
                        if validated_success:
                            TF=clip(TF+0.010*q_use*(1-TF),0.20,0.95); I=clip(I+0.008*q_use*(1-I),0.20,0.95)
                        if mi==4: provisional_debt=max(0.0,provisional_debt-0.02)
                    else:
                        false_consol+=1; I=clip(I-0.008,0.20,0.95); A=clip(A-0.006,0.20,0.95); H=clip(H+0.008,0.10,0.90)
                        if mi==4: provisional_debt+=0.06
            if mi==4:
                evidence_quality=0.55*validation_score+0.45*(1 if normative_correct else 0)
                if evidence_quality<0.42 and provisional_debt>0:
                    reverse_amt=min(provisional_debt,0.18)
                    I=clip(I+0.5*reverse_amt*(0.34-I),0.20,0.95); A=clip(A+0.5*reverse_amt*(0.40-A),0.20,0.95); H=clip(H-0.3*reverse_amt*H,0.10,0.90); provisional_debt=max(0.0,provisional_debt-reverse_amt)
            if normative_error: H=clip(H+0.004*(1-H),0.10,0.90)
        n+=1
        if (t+1)%phase_length==0:
            rec=tp/max(1,tp+fn); far=fp/n; vr=viol/max(1,crit); c=cap/max(1,tev); tr=ptraj/n
            phase_rec[pi]=rec; phase_far[pi]=far; phase_vr[pi]=vr; phase_cap[pi]=c; phase_traj[pi]=tr; phase_q[pi]=q_use; total_traj+=ptraj
    trap_cap=trap_capture/max(1,trap_events); false_rate=false_consol/max(1,consol_attempts); recovery=phase_rec[4]-phase_rec[3]
    final_rec=phase_rec[5]; final_far=phase_far[5]; final_vr=phase_vr[5]; final_q=phase_q[5]
    integrated=0.24*final_rec+0.16*(1-final_far)+0.18*(1-final_vr)+0.14*(1-trap_cap)+0.12*(1-false_rate)+0.08*clip(recovery+0.5,0,1)+0.08*clip((total_traj/episodes+1)/2,0,1)
    return trap_cap,false_rate,max(0.0,0.80-phase_q[3]),recovery,final_rec,final_far,final_vr,total_traj/episodes,integrated,final_q

# warm compile
run_one(1,0,60,10,phase_reward,phase_world,phase_false,phase_trap)
rows=[]
for seed in SPEC['seeds']:
    for mi,model in enumerate(SPEC['models']):
        vals=run_one(seed,mi,SPEC['episodes'],SPEC['phase_length'],phase_reward,phase_world,phase_false,phase_trap)
        rows.append((seed,model)+vals)
cols=['seed','model','trap_phase_capture_rate','trap_phase_false_consolidation_rate','post_trap_quality_damage','recovery_gain','final_holdout_recall','final_holdout_false_alarm','final_holdout_violation_rate','trajectory_value','integrated_score','final_quality']
df=pd.DataFrame(rows,columns=cols); df.to_csv(BASE/'seed_results_v277r1.csv',index=False)
summary=df.groupby('model').agg({c:'mean' for c in cols[2:]}).reset_index(); summary.to_csv(BASE/'summary_v277r1.csv',index=False)
S=summary.set_index('model'); time='simple_time_schedule'; suc='simple_success_driven'; u='tdc_unvalidated'; v='tdc_evidence_validated'; r='tdc_validated_with_reversal'
strong_simple=max(S.loc[time,'integrated_score'],S.loc[suc,'integrated_score']); best_tdc=max(S.loc[v,'integrated_score'],S.loc[r,'integrated_score'])
checks={
'validated_lower_capture_than_time':bool(S.loc[v,'trap_phase_capture_rate']<=S.loc[time,'trap_phase_capture_rate']-0.10),
'validated_lower_false_consolidation_than_unvalidated':bool(S.loc[v,'trap_phase_false_consolidation_rate']<=S.loc[u,'trap_phase_false_consolidation_rate']-0.12),
'reversal_best_trap_resistance':bool(S.loc[r,'trap_phase_capture_rate']<=S['trap_phase_capture_rate'].min()+1e-12),
'reversal_recovers_better':bool(S.loc[r,'recovery_gain']>=S.loc[v,'recovery_gain']+0.03),
'validated_better_final_recall_than_time':bool(S.loc[v,'final_holdout_recall']>=S.loc[time,'final_holdout_recall']+0.03),
'validated_lower_final_violations_than_time':bool(S.loc[v,'final_holdout_violation_rate']<=S.loc[time,'final_holdout_violation_rate']-0.03),
'complexity_earns_keep':bool(best_tdc>=strong_simple+0.03)}
(BASE/'acceptance_v277r1.json').write_text(json.dumps({'checks':checks,'best_tdc_margin_vs_best_simple':float(best_tdc-strong_simple)},indent=2))
print(summary.to_string(index=False)); print('\nACCEPTANCE');
for k,vv in checks.items(): print(f'{k}: {vv}')
print(f'passed={sum(checks.values())}/{len(checks)}'); print('Best TDC margin vs best simple:',best_tdc-strong_simple)
