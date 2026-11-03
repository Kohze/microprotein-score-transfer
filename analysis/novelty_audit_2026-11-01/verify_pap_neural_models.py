"""Check adapted Keras inference, shuffle means and every saved rank metric."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
B=P.parent/'stability_benchmark_2026-10-31'
O=P/'pap_neural_comparison'
sys.path.insert(0,str(B))
import run_benchmark as b
import hashlib
import json
import numpy as np
import pandas as pd
from pap_neural_runtime import PublishedPAP
from freeze_pap_neural_predictions import shuffled


def rho(y,p):
    yy,pp=pd.Series(np.asarray(y)),pd.Series(np.asarray(p))
    keep=yy.notna()&pp.notna()
    if keep.sum()<3 or yy[keep].std()<1e-14 or pp[keep].std()<1e-14:
        return np.nan
    return float(yy[keep].rank(method='average').corr(pp[keep].rank(method='average')))


def main():
    inputs=pd.read_parquet(O/'frozen_input_scores.parquet')
    lengths=np.array(list(map(len,inputs.index)))
    numerical=[]
    shuffle_errors=[]
    shuffle_checked=0
    for architecture in ['cnn2w1','cnn2w7']:
        model=PublishedPAP(architecture)
        for length in sorted(set(lengths)):
            seqs=inputs.index[lengths==length].tolist()
            for start in range(0,len(seqs),1024):
                part=seqs[start:start+1024]
                expected=model.numpy_predict(part)
                observed=inputs.loc[part,[architecture+'__'+c for c in ['full','internal','terminal']]].to_numpy()
                numerical.append(float(np.max(abs(expected-observed))))
            if architecture=='cnn2w7':
                # Fixed across-length sample, including larger samples from
                # the two target-length strata, independent of measurements.
                count=96 if length in [30,58] else 4
                step=max(1,len(seqs)//count)
                sample=seqs[::step][:count]
                for seed,component in [(20261105,'shuffle1_mean'),(20261106,'shuffle2_mean')]:
                    seqshuf=[z for s in sample for z in shuffled(s,seed)]
                    values=model.numpy_predict(seqshuf)[:,0].reshape(len(sample),19).mean(axis=1)
                    shuffle_errors.append(float(np.max(abs(values-inputs.loc[sample,architecture+'__'+component]))))
                    shuffle_checked+=len(sample)*19
    assert max(numerical)<1e-5 and max(shuffle_errors)<1e-5
    pred=pd.read_parquet(O/'frozen_predictions.parquet').set_index('sequence')
    maps=json.loads((O/'SOURCE_ONLY_CALIBRATIONS.json').read_text())
    calibration_errors=[]
    for key,info in maps.items():
        expected=np.interp(pred[key],info['x_thresholds'],info['y_thresholds'])
        calibration_errors.append(float(np.max(abs(expected-pred[key+'__source_WT_calibrated']))))
    # sklearn preserves float32 input/output here; NumPy interp evaluates the
    # same stored threshold map in float64. Allow two float32 ulps on the unit
    # source-assay scale, rather than requiring impossible float64 equality.
    calibration_tolerance=2*np.finfo(np.float32).eps
    assert max(calibration_errors)<calibration_tolerance
    native=pd.read_parquet(P/'independent_test/evaluated_external_predictions.parquet')
    native=native.merge(pred,left_on='aa',right_index=True,validate='one_to_one')
    guo=b.A[b.A.group.eq('microprotein_WT')&b.A.mean_stability.notna()&~b.A.inconsistent_parent_id]
    guo=guo.merge(pred,left_on='protein_sequence',right_index=True,validate='many_to_one')
    additional=pd.read_parquet(B/'results/additional_transfer.parquet')
    additional=additional[additional.strict_disjoint].merge(pred,left_on='protein_sequence',right_index=True,validate='one_to_one')
    metrics=pd.read_csv(O/'native_metrics.csv')
    errors=[]
    for row in metrics.itertuples():
        if row.cohort.startswith('Voutsinos_'):
            label=row.cohort.removeprefix('Voutsinos_')
            z=native[native[label+'_eligible']]; y=z.abundance_score
        elif row.cohort=='Guo_additional_strict602':
            z=additional;y=z.mean_stability
        else:
            z=guo[guo.source_sheet.eq(row.cohort.removeprefix('Guo_'))];y=z.mean_stability
        assert len(z)==row.n
        errors.append(abs(rho(y,z[row.model])-row.rho))
    paired=pd.read_parquet(P/'independent_interventions/evaluated_predictions.parquet')
    delta=pd.read_parquet(O/'paired_delta_predictions.parquet')
    assert paired.source_excel_row.equals(delta.source_excel_row)
    effectmetrics=pd.read_csv(O/'paired_effect_metrics.csv')
    for row in effectmetrics.itertuples():
        subset=row.cohort.removeprefix('Chu_')
        if subset.endswith('_current_transcript_match'):
            mask=paired.primary60_common_eligible&paired.WT_matches_current_transcript_C58
        else:
            mask=paired[subset+'_eligible']
        assert mask.sum()==row.n
        wt=pred.loc[paired.wt_sequence,row.model].to_numpy()
        rt=pred.loc[paired.variant_sequence,row.model].to_numpy()
        assert np.max(abs(rt-wt-delta[row.model]))<1e-12
        errors.append(abs(rho(paired.loc[mask,'observed_delta'],delta.loc[mask,row.model])-row.delta_rho))
        if np.isfinite(row.native_wt_rho):
            errors.append(abs(rho(paired.loc[mask,'wt_mean'],wt[mask])-row.native_wt_rho))
            errors.append(abs(rho(paired.loc[mask,'rt_mean'],rt[mask])-row.native_rt_rho))
    assert max(errors)<1e-12
    freeze=json.loads((O/'PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_predictions.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    receipt=dict(status='passed',published_graph='preserved; backend import and legacy HDF5 loading adapted only',
        all_native_input_scores_independently_checked=len(inputs)*2,native_score_max_NumPy_error=max(numerical),
        shuffled_scores_independently_checked=shuffle_checked,shuffle_mean_max_NumPy_error=max(shuffle_errors),
        calibration_threshold_maps_reconstructed=len(maps),calibration_max_error=max(calibration_errors),
        calibration_tolerance=float(calibration_tolerance),
        calibration_precision='stored sklearn float32 versus independent NumPy float64 interpolation',
        native_metric_rows_recalculated=len(metrics),paired_metric_rows_recalculated=len(effectmetrics),
        metric_max_rank_error=max(errors),all_pair_delta_values_reconstructed=True,frozen_predictions_unchanged=True,
        limits='Numerical/implementation verification; does not establish biology or retraining uncertainty')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    with b.threadpool_limits(limits=4):
        main()
