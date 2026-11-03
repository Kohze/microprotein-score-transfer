"""Independently reconstruct all extended deltas and rank statistics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'

def rho(y,p):
    a,c=pd.Series(np.asarray(y)),pd.Series(np.asarray(p))
    if a.std()<1e-14 or c.std()<1e-14:return np.nan
    return float(a.rank().corr(c.rank()))

def main():
    d=pd.read_parquet(O/'evaluated_window_predictions.parquet')
    original=pd.read_parquet(O/'evaluated_predictions.parquet')
    assert d.source_excel_row.equals(original.source_excel_row)
    for column in original:assert d[column].equals(original[column]),column
    score=pd.read_parquet(O/'frozen_window_controls.parquet').set_index('C23_sequence')
    large=pd.read_parquet(P/'pap_neural_comparison/frozen_predictions.parquet').set_index('sequence')
    metrics=pd.read_csv(O/'window_control_metrics.csv');errors=[]
    for row in metrics.itertuples():
        if '__C23__' in row.model:
            wt=score.loc[d.wt_sequence.str[-23:],row.model].to_numpy();rt=score.loc[d.variant_sequence.str[-23:],row.model].to_numpy()
        else:
            wt=large.loc[d.wt_sequence,row.model].to_numpy();rt=large.loc[d.variant_sequence,row.model].to_numpy()
        assert np.max(abs(wt-d[row.model+'__wt']))==0
        assert np.max(abs(rt-d[row.model+'__rt']))==0
        delta=rt-wt
        assert np.max(abs(delta-d[row.model+'__delta']))==0
        keep=d[row.cohort+'_eligible'];z=d[keep]
        assert keep.sum()==row.n and z.target_dependence_group.nunique()==row.groups
        for actual,saved in [(rho(z.wt_mean,wt[keep]),row.native_WT_rho),(rho(z.rt_mean,rt[keep]),row.native_RT_rho),
            (rho(z.observed_delta,delta[keep]),row.delta_rho),(rho(z.observed_delta_r1,delta[keep]),row.replicate1_delta_rho),
            (rho(z.observed_delta_r2,delta[keep]),row.replicate2_delta_rho)]:
            assert np.isnan(actual)==np.isnan(saved)
            if np.isfinite(actual):errors.append(abs(actual-saved))
    assert max(errors)<1e-12
    freeze=json.loads((O/'WINDOW_CONTROL_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_window_controls.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    result=dict(status='passed',extended_metric_rows_recalculated=len(metrics),max_rank_error=max(errors),
        all_C23_C30_endpoint_scores_and_deltas_reconstructed=True,original48_rows_and_all_pairs_preserved=True,
        frozen_hash_unchanged=True,all6604_C23_network_inputs_NumPy_checked_at_freeze=True,
        scope='Post-hoc matched-input comparison; numerical verification does not imply architecture causation or full exposure independence')
    (O/'WINDOW_CONTROL_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
