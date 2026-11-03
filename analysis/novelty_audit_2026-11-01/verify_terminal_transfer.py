"""Independently check source-screen joins, all metrics and original inference."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'
sys.path.insert(0,str(P.parent/'stability_benchmark_2026-10-31'))
import run_benchmark as b
sys.path.insert(0,str(O/'vendor'))
import hashlib
import json
import runpy
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from catboost import CatBoostRegressor
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def rho(y,p):
    a,c=pd.Series(np.asarray(y)),pd.Series(np.asarray(p))
    if a.std()<1e-14 or c.std()<1e-14:return np.nan
    return float(a.rank().corr(c.rank()))

def main():
    workbook=load_workbook(O/'Hickey2021_STable5.xlsx',read_only=True,data_only=True)
    rows=list(workbook['(A) All C-terminal 23mers'].iter_rows(min_row=6,values_only=True))
    assert len(rows)==22564
    screen={row[11][:-1] for row in rows}
    assert len(screen)==22364 and all(row[11].endswith('*') and row[11][:-1]==row[12] for row in rows)
    symbols={str(row[1]).upper() for row in rows if row[1] is not None}
    original=pd.read_parquet(P/'independent_interventions/evaluated_predictions.parquet')
    original=original[original.primary60_common_eligible].set_index('source_excel_row')
    audit=pd.read_parquet(O/'Chu_C23_exposure_audit.parquet')
    audit=audit[audit.primary60_common_eligible]
    for row in audit.itertuples():
        pair=original.loc[row.source_excel_row]
        wm=pair.wt_sequence[-23:] in screen;rm=pair.variant_sequence[-23:] in screen
        assert wm==row.WT_exact_development_screen_match and rm==row.RT_exact_development_screen_match
        assert (wm or rm)==row.either_endpoint_screen_match
        assert (str(pair.gene_name).upper() in symbols)==row.available_gene_symbol_screen_overlap
    if '--exposure-only' in sys.argv:
        result=dict(status='passed',original_workbook_rows_independently_read=len(rows),
            unique_screen_C23=len(screen),all1743_endpoint_and_symbol_joins_rechecked=True,
            confirmed90_10_training_split_membership=False,target_model_associations_computed=False)
        (O/'INDEPENDENT_EXPOSURE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return
    pred=pd.read_parquet(O/'frozen_predictions.parquet').set_index('C23_sequence').DEGRONOPEDIA_C23
    model=CatBoostRegressor();model.load_model(str(O/'published/models/c-terminus.cbm'))
    functions=runpy.run_path(str(O/'published/calculate-desc.py'),run_name='original_functions_verification')
    sample=original.sort_index().iloc[::max(1,len(original)//16)].iloc[:16]
    errors=[]
    # Pass the ORIGINAL58mer to the official descriptor preparation, unlike
    # the freeze's C23 input: verify complete-sequence terminal extraction.
    for row in sample.itertuples():
        for sequence in [row.wt_sequence,row.variant_sequence]:
            fragments=functions['generate_ogryzki'](sequence)[0]
            ff=functions['generate_ML_tsv'](sequence[-23:],fragments)[model.feature_names_]
            expected=float(model.predict(ff,thread_count=4)[0]);errors.append(abs(expected-pred.loc[sequence[-23:]]))
    assert max(errors)<1e-12
    evaluated=pd.read_parquet(O/'evaluated_predictions.parquet')
    metrics=pd.read_csv(O/'transfer_metrics.csv');metric_errors=[]
    for row in metrics.itertuples():
        z=evaluated[evaluated[row.cohort+'_eligible']]
        assert len(z)==row.n and z.target_dependence_group.nunique()==row.groups
        wt=z[row.model+'__wt'];rt=z[row.model+'__rt'];delta=rt-wt
        assert np.max(abs(delta-z[row.model+'__delta']))<1e-12
        for saved,actual in [(row.native_WT_rho,rho(z.wt_mean,wt)),(row.native_RT_rho,rho(z.rt_mean,rt)),
            (row.delta_rho,rho(z.observed_delta,delta)),(row.replicate1_delta_rho,rho(z.observed_delta_r1,delta)),
            (row.replicate2_delta_rho,rho(z.observed_delta_r2,delta))]:
            assert np.isnan(saved)==np.isnan(actual)
            if np.isfinite(saved):metric_errors.append(abs(saved-actual))
    assert max(metric_errors)<1e-12
    freeze=json.loads((O/'PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_predictions.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    result=dict(status='passed',original_workbook_rows_independently_read=len(rows),exact_screen_memberships_rechecked=1743,
        complete58mer_original_descriptor_predictions_checked=len(errors),original_prediction_max_error=max(errors),
        metric_rows_recalculated=len(metrics),rank_metric_max_error=max(metric_errors),all_pair_deltas_reconstructed=True,
        frozen_hash_unchanged=True,limitations='Numerical/source-join verification; does not certify unavailable individual-training membership, exhaustive aliases or biology')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    with b.threadpool_limits(limits=4):main()
