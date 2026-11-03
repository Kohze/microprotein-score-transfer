"""Evaluate the frozen published default/order-capable PAP models and controls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P=Path(__file__).resolve().parent
O=P/'pap_neural_comparison'
B=P.parent/'stability_benchmark_2026-10-31'


def rho(y,p):
    y,p=np.asarray(y),np.asarray(p)
    valid=np.isfinite(y)&np.isfinite(p)
    if valid.sum()<3 or np.std(y[valid])<1e-14 or np.std(p[valid])<1e-14:
        return np.nan
    return float(spearmanr(y[valid],p[valid]).statistic)


def bootstrap(y,group,predictions,comparisons,seed):
    members=[np.flatnonzero(group==g) for g in np.unique(group)]
    rng=np.random.default_rng(seed)
    values={m:[] for m in predictions}
    for draw in range(1000):
        ix=np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
        for model,p in predictions.items():
            values[model].append(rho(y[ix],p[ix]))
        if (draw+1)%250==0:
            print('conditional bootstrap',seed,draw+1,flush=True)
    out=[]
    for a,b,role in comparisons:
        gap=np.array(values[a])-np.array(values[b])
        lo,hi=np.nanquantile(gap,[.025,.975])
        out.append(dict(contrast=a+' minus '+b,role=role,n=len(y),groups=len(members),
            estimate=rho(y,predictions[a])-rho(y,predictions[b]),low=lo,high=hi,
            scope='conditional on frozen/source-only predictions; no retraining or multiplicity-adjusted inference'))
    return out


def main():
    freeze=json.loads((O/'PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_predictions.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    pred=pd.read_parquet(O/'frozen_predictions.parquet')
    models=[c for c in pred if c!='sequence']
    target=pd.read_parquet(P/'independent_test/evaluated_external_predictions.parquet')
    native=target.merge(pred,left_on='aa',right_on='sequence',validate='one_to_one')
    assert len(native)==len(target)
    metrics=[]
    for subset in ['primary60','sensitivity40']:
        z=native[native[subset+'_eligible']]
        for model in models:
            metrics.append(dict(cohort='Voutsinos_'+subset,model=model,n=len(z),groups=int(z.protein_id.nunique()),
                rho=rho(z.abundance_score,z[model]),provenance='author-held-out target-study-trained reference'))
    source=pd.read_parquet(B/'natural_assay_records.parquet')
    source=source[source.source_sheet.isin(['GFP_MP','MP_mCherry'])&source.group.eq('microprotein_WT')&
        source.mean_stability.notna()&~source.inconsistent_parent_id].merge(pred,left_on='protein_sequence',right_on='sequence',validate='many_to_one')
    additional=pd.read_parquet(B/'results/additional_transfer.parquet')
    additional=additional[additional.strict_disjoint].merge(pred,left_on='protein_sequence',right_on='sequence',validate='one_to_one')
    assert len(additional)==602
    for label,z in [('Guo_GFP_MP',source[source.source_sheet.eq('GFP_MP')]),
                    ('Guo_MP_mCherry',source[source.source_sheet.eq('MP_mCherry')]),('Guo_additional_strict602',additional)]:
        for model in models:
            if model.endswith('__source_WT_calibrated'):
                continue # avoid presenting source calibration resubstitution as held-out prediction
            metrics.append(dict(cohort=label,model=model,n=len(z),rho=rho(z.mean_stability,z[model]),
                provenance='published frozen model; exact source-genome/pretraining separation not exhaustive'))
    pd.DataFrame(metrics).to_csv(O/'native_metrics.csv',index=False)
    paired=pd.read_parquet(P/'independent_interventions/evaluated_predictions.parquet')
    delta=pd.DataFrame(index=paired.index)
    indexed=pred.set_index('sequence')
    effects=[]
    for model in models:
        wt=indexed.loc[paired.wt_sequence,model].to_numpy()
        rt=indexed.loc[paired.variant_sequence,model].to_numpy()
        delta[model]=rt-wt
        for subset in ['primary60_common','sensitivity40_common','primary60_head_only','sensitivity40_head_only']:
            mask=paired[subset+'_eligible']
            effects.append(dict(cohort='Chu_'+subset,model=model,n=int(mask.sum()),
                groups=int(paired.loc[mask,'target_dependence_group'].nunique()),
                native_wt_rho=rho(paired.loc[mask,'wt_mean'],wt[mask]),native_rt_rho=rho(paired.loc[mask,'rt_mean'],rt[mask]),
                delta_rho=rho(paired.loc[mask,'observed_delta'],delta.loc[mask,model]),
                delta_r1_rho=rho(paired.loc[mask,'observed_delta_r1'],delta.loc[mask,model]),
                delta_r2_rho=rho(paired.loc[mask,'observed_delta_r2'],delta.loc[mask,model]),
                predicted_zero_delta_fraction=float(np.mean(abs(delta.loc[mask,model])<1e-14))))
        mask=paired.primary60_common_eligible&paired.WT_matches_current_transcript_C58
        effects.append(dict(cohort='Chu_primary60_common_current_transcript_match',model=model,n=int(mask.sum()),
            groups=int(paired.loc[mask,'target_dependence_group'].nunique()),delta_rho=rho(paired.loc[mask,'observed_delta'],delta.loc[mask,model])))
    delta.insert(0,'source_excel_row',paired.source_excel_row)
    delta.to_parquet(O/'paired_delta_predictions.parquet',index=False)
    pd.DataFrame(effects).to_csv(O/'paired_effect_metrics.csv',index=False)
    comparisons=[]
    for architecture in ['cnn2w1','cnn2w7']:
        for representation in ['C30','full_cargo']:
            base=architecture+'__'+representation+'__full'
            comparisons.append((base+'__source_WT_calibrated',base,'source-only calibration'))
    for representation in ['C30','full_cargo']:
        base='cnn2w7__'+representation+'__'
        comparisons.extend([(base+'full',base+'shuffle1_mean','order outside fixed ends'),
                            (base+'full',base+'shuffle2_mean','second-shuffle sensitivity')])
    keep=paired.primary60_common_eligible.to_numpy()
    needed=sorted({m for a,b,_ in comparisons for m in [a,b]})
    points={m:delta.loc[keep,m].to_numpy() for m in needed}
    contrasts=bootstrap(paired.loc[keep,'observed_delta'].to_numpy(),paired.loc[keep,'target_dependence_group'].to_numpy(),
                        points,comparisons,20261107)
    pd.DataFrame(contrasts).to_csv(O/'paired_effect_contrasts.csv',index=False)
    z=native[native.primary60_eligible]
    # Both representations are exactly the same30mer inputs on this cohort.
    for architecture in ['cnn2w1','cnn2w7']:
        assert np.array_equal(z[architecture+'__C30__full'],z[architecture+'__full_cargo__full'])
    native_compare=[('cnn2w7__C30__full','cnn2w7__C30__shuffle1_mean','order outside fixed ends'),
                    ('cnn2w7__C30__full','cnn2w7__C30__shuffle2_mean','second-shuffle sensitivity')]
    pp={m:z[m].to_numpy() for m in sorted({m for a,b,_ in native_compare for m in [a,b]})}
    contrasts_native=bootstrap(z.abundance_score.to_numpy(),z.protein_id.to_numpy(),pp,native_compare,20261108)
    pd.DataFrame(contrasts_native).to_csv(O/'native_order_contrasts.csv',index=False)
    receipt=dict(status='neural baselines and order controls evaluated; independent metric verification pending',
        target_model_training=False,calibration_only_on_Guo_GFP_WT=True,
        primary_native_n=14143,primary_paired_n=1743,primary_paired_groups=1591,
        prediction_freeze_sha256=hashlib.sha256((O/'PREDICTION_FREEZE.json').read_bytes()).hexdigest(),
        model1_order_control='exact analytic equality; sampled numerical check',
        model7_order_control='two independent19-shuffle means; not measured shuffled-peptide outcomes',
        source_calibration_resubstitution_not_reported=True,
        limitations=['Target-study-trained native references are not equally external competitors','Conditional intervals, not retraining uncertainty','No causal order/terminal interpretation'])
    (O/'EVALUATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(metrics).query("cohort=='Voutsinos_primary60' and model.str.endswith('__full')",engine='python').to_string(index=False),flush=True)
    print(pd.DataFrame(effects).query("cohort=='Chu_primary60_common' and (model.str.endswith('__full') or model.str.endswith('__full__source_WT_calibrated'))",engine='python').to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts_native).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
