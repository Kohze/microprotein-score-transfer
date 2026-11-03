"""Evaluate frozen source-only predictions on paired independent reporter changes."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

P=Path(__file__).resolve().parent
O=P/'independent_interventions'
D=P/'independent_data/Chu2026_supplements'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def rho(y,p):
    y,p=np.asarray(y),np.asarray(p)
    valid=np.isfinite(y)&np.isfinite(p)
    if valid.sum()<3 or np.std(p[valid])<1e-14 or np.std(y[valid])<1e-14:
        return np.nan
    return float(spearmanr(y[valid],p[valid]).statistic)


def main():
    freeze=json.loads((O/'PREDICTION_FREEZE.json').read_text())
    separation=json.loads((O/'SOURCE_SEPARATION_AUDIT.json').read_text())
    assert freeze['protocol_sha256']==separation['protocol_sha256']==sha(O/'COMPARISON_PROTOCOL.txt')
    assert freeze['prediction_sha256']==sha(O/'frozen_predictions.parquet')
    d=pd.read_parquet(O/'separated_sequence_pairs.parquet')
    sheet=load_workbook(D/'41467_2026_68979_MOESM1_ESM.xlsx',read_only=True,data_only=True).active
    rows=[]
    fields=['released_delta','released_wt_mean','released_rt_mean','wt_count_r1','rt_count_r1',
            'wt_r1','rt_r1','wt_count_r2','rt_count_r2','wt_r2','rt_r2']
    for rownum,values in enumerate(sheet.iter_rows(min_row=5,values_only=True),start=5):
        rows.append(dict(source_excel_row=rownum,**dict(zip(fields,values[9:20]))))
    outcomes=pd.DataFrame(rows)
    for field in fields:
        outcomes[field]=pd.to_numeric(outcomes[field],errors='coerce')
    d=d.merge(outcomes,on='source_excel_row',validate='one_to_one')
    assert len(d)==3249
    d['wt_mean']=(d.wt_r1+d.wt_r2)/2
    d['rt_mean']=(d.rt_r1+d.rt_r2)/2
    d['observed_delta_r1']=d.rt_r1-d.wt_r1
    d['observed_delta_r2']=d.rt_r2-d.wt_r2
    d['observed_delta']=(d.observed_delta_r1+d.observed_delta_r2)/2
    scorecols=['wt_r1','wt_r2','rt_r1','rt_r2']
    countcols=['wt_count_r1','wt_count_r2','rt_count_r1','rt_count_r2']
    d['complete_qualified_measurements']=d[scorecols].notna().all(axis=1)&d[countcols].ge(60).all(axis=1)
    qualified=d[d.complete_qualified_measurements]
    assert qualified[scorecols].ge(1-1e-8).all().all() and qualified[scorecols].le(8+1e-8).all().all()
    consistency={field:float(abs(d[field]-d[computed]).max()) for field,computed in [
        ('released_wt_mean','wt_mean'),('released_rt_mean','rt_mean'),('released_delta','observed_delta')]}
    pred=pd.read_parquet(O/'frozen_predictions.parquet').set_index('sequence')
    models=[c for c in pred if not c.endswith('__source_range_clipped')]
    for model in models:
        d[model+'__wt']=pred.loc[d.wt_sequence,model].to_numpy()
        d[model+'__rt']=pred.loc[d.variant_sequence,model].to_numpy()
        d[model+'__delta']=d[model+'__rt']-d[model+'__wt']
    for label in ['primary60','sensitivity40']:
        for cohort in ['head_only','common']:
            key=label+'_'+cohort+'_eligible'
            d[key]=d[key]&d.complete_qualified_measurements
    d.to_parquet(O/'evaluated_predictions.parquet',index=False)
    d.to_csv(O/'evaluated_predictions.csv',index=False)
    metrics=[]
    def evaluate(z,label):
        for model in models:
            delta=z[model+'__delta'].to_numpy()
            effect=z.observed_delta.to_numpy()
            called=abs(effect)>=1
            metrics.append(dict(cohort=label,model=model,n=len(z),
                groups=int(z.target_dependence_group.nunique()),
                native_wt_rho=rho(z.wt_mean,z[model+'__wt']),native_rt_rho=rho(z.rt_mean,z[model+'__rt']),
                delta_rho=rho(effect,delta),delta_r1_rho=rho(z.observed_delta_r1,delta),
                delta_r2_rho=rho(z.observed_delta_r2,delta),predicted_zero_delta_fraction=float(np.mean(abs(delta)<1e-14)),
                called_effects_n=int(called.sum()),
                sign_accuracy_called=float(((np.sign(effect[called])==np.sign(delta[called])).astype(float)+.5*(delta[called]==0)).mean()) if called.any() else np.nan,
                sign_AUROC_called=float(roc_auc_score(effect[called]>0,delta[called])) if len(np.unique(effect[called]>0))==2 else np.nan))
    for label in ['primary60_common','sensitivity40_common','primary60_head_only','sensitivity40_head_only']:
        evaluate(d[d[label+'_eligible']],label)
    primary=d[d.primary60_common_eligible].reset_index(drop=True)
    evaluate(primary[primary.WT_matches_current_transcript_C58],'primary60_common_current_transcript_match')
    for label,mask in [('RT_length_le20',primary.reported_RT_length.le(20)),
                       ('RT_length_21to57',primary.reported_RT_length.between(21,57)),
                       ('RT_length_ge58',primary.reported_RT_length.ge(58))]:
        evaluate(primary[mask],label)
    pd.DataFrame(metrics).to_csv(O/'transfer_metrics.csv',index=False)
    print(pd.DataFrame(metrics).query("cohort=='primary60_common'")[['model','n','groups','native_wt_rho','native_rt_rho','delta_rho']].to_string(index=False),flush=True)
    comparisons=[('GFP_MP__T','GFP_MP__A','primary fitted contrast'),
        ('MP_mCherry__T','MP_mCherry__A','context sensitivity'),
        ('PAP_C30__source_WT_calibrated','PAP_C30','secondary calibration'),
        ('PAP_full_cargo__source_WT_calibrated','PAP_full_cargo','secondary calibration'),
        ('PAP_full_cargo','PAP_full_cargo_composition','secondary terminal component'),
        ('PAP_C30','PAP_C30_composition','secondary terminal component')]
    groups=primary.target_dependence_group.to_numpy()
    members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    rng=np.random.default_rng(20261104)
    yy=primary.observed_delta.to_numpy()
    needed=sorted({m for a,b,_ in comparisons for m in [a,b]})
    predictions={m:primary[m+'__delta'].to_numpy() for m in needed}
    boot={m:[] for m in needed}
    for draw in range(1000):
        ix=np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
        for model,p in predictions.items():
            boot[model].append(rho(yy[ix],p[ix]))
        if (draw+1)%250==0:
            print('paired operational-group bootstrap',draw+1,flush=True)
    contrasts=[]
    for a,b,role in comparisons:
        gap=np.array(boot[a])-np.array(boot[b])
        lo,hi=np.nanquantile(gap,[.025,.975])
        contrasts.append(dict(contrast=a+' minus '+b,role=role,n=len(primary),groups=len(members),
            delta_rho_gap=rho(yy,predictions[a])-rho(yy,predictions[b]),low=lo,high=hi,
            scope='conditional on source-only predictions; no full retraining or multiplicity correction'))
    pd.DataFrame(contrasts).to_csv(O/'paired_delta_contrasts.csv',index=False)
    receipt=dict(status='independent intervention evaluation completed; independent numerical verification pending',
        target_pairs_released=len(d),complete_quality_pairs=int(d.complete_qualified_measurements.sum()),
        eligible_counts={c:int(d[c].sum()) for c in d if c.endswith('_eligible')},
        primary_groups=len(members),source_average_max_disagreement=consistency,
        primary_replicate_delta_rho=rho(primary.observed_delta_r1,primary.observed_delta_r2),
        prediction_freeze_sha256=sha(O/'PREDICTION_FREEZE.json'),target_outcome_fitting=False,
        prediction_direction_reversed_after_target_inspection=False,
        biological_scope='Independent canonical C-terminal 58mer sequence-replacement reporter pairs; not native nORF function')
    (O/'EVALUATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
