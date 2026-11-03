"""Independently refit heads, reconstruct baseline scores and recalculate paired metrics."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
B=P.parent/'stability_benchmark_2026-10-31'
O=P/'independent_interventions'
sys.path.insert(0,str(B))
import run_benchmark as b
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from freeze_published_baselines import literal_model


def rank_corr(y,p):
    yy,pp=pd.Series(np.asarray(y)),pd.Series(np.asarray(p))
    if pp.std()<1e-14 or yy.std()<1e-14:
        return np.nan
    return float(yy.rank(method='average').corr(pp.rank(method='average')))


def main():
    pred=pd.read_parquet(O/'frozen_predictions.parquet')
    ff=pd.read_parquet(O/'features.parquet')
    specs=b.SETS
    errors=[]
    fits=json.loads((O/'SOURCE_ONLY_FITS.json').read_text())
    for record in fits:
        context,model=record['source'],record['model']
        d=b.A[b.A.source_sheet.eq(context)&b.A.group.eq('microprotein_WT')&b.A.mean_stability.notna()&~b.A.inconsistent_parent_id]
        x=ff.loc[d.protein_sequence,specs[model]].to_numpy()
        z=ff.loc[pred.sequence,specs[model]].to_numpy()
        scale=StandardScaler().fit(x)
        fit=Ridge(alpha=record['alpha'],solver='cholesky').fit(scale.transform(x),d.mean_stability)
        errors.append(float(np.max(abs(fit.predict(scale.transform(z))-pred[context+'__'+model]))))
    assert max(errors)<1e-8
    param=json.loads((B/'inputs/PAP_HUMAN30_PARSED_COEFFICIENTS.json').read_text())
    q=literal_model(P/'published_predictor_comparison/QCDpred.py','model')
    raw_errors=[]
    for row in pred.itertuples():
        s=row.sequence
        comp=sum(s.count(a)*v for a,v in param['comp'].items())*30/len(s)
        tail=s[-30:]
        ccomp=sum(tail.count(a)*v for a,v in param['comp'].items())
        term=0.
        for family in ['pssm','pair','triplet']:
            for motif,value in param[family].items():
                if all(s[-int(item[1:])]==item[0] for item in motif.split(':')):
                    term+=value
        values=[comp,ccomp,param['intercept']+comp+term,param['intercept']+ccomp+term,term]
        saved=[row.PAP_full_cargo_composition,row.PAP_C30_composition,row.PAP_full_cargo,row.PAP_C30,row.PAP_terminal]
        raw_errors.extend(abs(np.array(values)-np.array(saved)))
        weights=np.array([q[a] for a in s])
        expected=1-float(expit(q['intersect']+np.convolve(weights,np.ones(17),mode='valid')).mean())
        raw_errors.append(abs(expected-row.QCDPred_mean17))
    assert max(raw_errors)<1e-12
    # Reconstruct every saved calibration by its explicit threshold arrays.
    cal=json.loads((O/'SOURCE_ONLY_CALIBRATIONS.json').read_text())
    cal_errors=[]
    for model,info in cal.items():
        values=np.interp(pred[model],info['x_thresholds'],info['y_thresholds'])
        cal_errors.append(float(np.max(abs(values-pred[model+'__source_WT_calibrated']))))
    assert max(cal_errors)<1e-12
    evaluated=pd.read_parquet(O/'evaluated_predictions.parquet')
    subsets={key:evaluated[evaluated[key+'_eligible']] for key in [
        'primary60_common','sensitivity40_common','primary60_head_only','sensitivity40_head_only']}
    primary=subsets['primary60_common']
    subsets['primary60_common_current_transcript_match']=primary[primary.WT_matches_current_transcript_C58]
    subsets['RT_length_le20']=primary[primary.reported_RT_length.le(20)]
    subsets['RT_length_21to57']=primary[primary.reported_RT_length.between(21,57)]
    subsets['RT_length_ge58']=primary[primary.reported_RT_length.ge(58)]
    metrics=pd.read_csv(O/'transfer_metrics.csv')
    metric_errors=[]
    for row in metrics.itertuples():
        z=subsets[row.cohort]
        assert len(z)==row.n and z.target_dependence_group.nunique()==row.groups
        wt,rt=z[row.model+'__wt'].to_numpy(),z[row.model+'__rt'].to_numpy()
        delta=rt-wt
        for field,actual in [
            ('native_wt_rho',rank_corr(z.wt_mean,wt)),('native_rt_rho',rank_corr(z.rt_mean,rt)),
            ('delta_rho',rank_corr(z.observed_delta,delta)),
            ('delta_r1_rho',rank_corr(z.observed_delta_r1,delta)),('delta_r2_rho',rank_corr(z.observed_delta_r2,delta))]:
            saved=getattr(row,field)
            assert np.isnan(actual)==np.isnan(saved)
            if np.isfinite(actual):
                metric_errors.append(abs(actual-saved))
    assert max(metric_errors)<1e-12
    assert not primary.source_gene_overlap.any() and not primary.PAP_training_gene_overlap.any()
    assert not primary.PAP_training_exact30_window.any() and not primary.human25_training_exact25_window.any()
    assert primary.wt_endswith_shared.all() and primary.reported_RT_length_consistent.all()
    assert all(primary[cols].ge(60).all() for cols in ['wt_count_r1','rt_count_r1','wt_count_r2','rt_count_r2'])
    freeze=json.loads((O/'PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_predictions.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    result=dict(status='passed',source_heads_independently_refit=len(fits),head_max_absolute_error=max(errors),
        independent_PAP_QCD_values_max_error=max(raw_errors),all_calibration_thresholds_reconstructed=True,
        calibration_max_error=max(cal_errors),metric_rows_independently_checked=len(metrics),
        max_rank_metric_error=max(metric_errors),primary_pairs=len(primary),primary_groups=int(primary.target_dependence_group.nunique()),
        all_primary_gene_exact_window_and_construct_checks=True,frozen_predictions_unchanged=True,
        source_assay_calibration_only=True,limits='Operational exclusions; conditional uncertainty, not causal or native nORF validation')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with b.threadpool_limits(limits=4):
        main()
