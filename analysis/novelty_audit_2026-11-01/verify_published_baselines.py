"""Independently check frozen baseline values, rank metrics and selection policies."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from freeze_published_baselines import literal_model

P=Path(__file__).resolve().parent
O=P/'published_predictor_comparison'
B=P.parent/'stability_benchmark_2026-10-31'


def rank_corr(y,p):
    yy,pp=pd.Series(np.asarray(y)),pd.Series(np.asarray(p))
    valid=yy.notna()&pp.notna()
    return float(yy[valid].rank(method='average').corr(pp[valid].rank(method='average')))


def main():
    freeze=json.loads((O/'BASELINE_PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_published_predictions.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    q=literal_model(O/'QCDpred.py','model')
    f=pd.read_parquet(O/'frozen_published_predictions.parquet')
    errors=[]
    for row in f.itertuples():
        if len(row.sequence)<17:
            assert np.isnan(row.QCDPred_mean17_stability)
            continue
        weights=np.array([q[a] for a in row.sequence])
        # Convolution-based calculation is independent of the original
        # scalar Python sum over each explicitly generated tile.
        prob=expit(q['intersect']+np.convolve(weights,np.ones(17),mode='valid'))
        expected=1-float(prob.mean())
        errors.append(abs(expected-row.QCDPred_mean17_stability))
    assert max(errors)<1e-12
    d=pd.read_parquet(O/'evaluated_published_predictions.parquet')
    points=pd.read_csv(O/'published_baseline_metrics.csv')
    metric_errors=[]
    guo=pd.read_parquet(B/'natural_assay_records.parquet')
    guo=guo[guo.group.eq('microprotein_WT')&guo.mean_stability.notna()&~guo.inconsistent_parent_id]
    guo=guo.merge(f,left_on='protein_sequence',right_on='sequence',validate='many_to_one')
    for row in points.itertuples():
        if row.cohort.startswith('Voutsinos_'):
            subset=row.cohort.removeprefix('Voutsinos_')
            z=d[d[subset+'_eligible']]
            y=z.abundance_score
        else:
            source=row.cohort.removeprefix('Guo_')
            z=guo[guo.source_sheet.eq(source)]
            y=z.mean_stability
        assert int(z[row.model].notna().sum())==row.n
        metric_errors.append(abs(rank_corr(y,z[row.model])-row.rho))
    assert max(metric_errors)<1e-12
    policies=pd.read_csv(O/'published_baseline_policies.csv')
    policy_errors=[]
    for row in policies.itertuples():
        z=d[d[row.subset+'_eligible']].sort_values('aa')
        selected=z.sort_values(row.model,ascending=False,kind='stable').head(row.selected)
        policy_errors.append(abs(float(selected.abundance_score.mean())-row.selected_mean_assay_score))
    assert max(policy_errors)<1e-12
    result=dict(status='passed',independent_convolution_checked_sequences=len(errors),
        max_QCDPred_value_error=max(errors),rank_metrics_recomputed=len(points),
        max_rank_metric_error=max(metric_errors),all_policy_point_means_recomputed=True,
        max_policy_error=max(policy_errors),frozen_prediction_hash_unchanged=True,
        training_rows_match_published_count=freeze['QCDPred_labeled_training_rows']==13559,
        scope='Point values/metrics and policy means verified; uncertainty is conditional and not retraining validation')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
