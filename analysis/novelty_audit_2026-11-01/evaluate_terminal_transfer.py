"""Evaluate frozen original C23 model on every declared exposure stratum."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from evaluate_pap_neural_models import rho,bootstrap

P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'

def main():
    freeze=json.loads((O/'PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_predictions.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    score=pd.read_parquet(O/'frozen_predictions.parquet').set_index('C23_sequence').DEGRONOPEDIA_C23
    pairs=pd.read_parquet(P/'independent_interventions/evaluated_predictions.parquet')
    pairs=pairs[pairs.primary60_common_eligible].copy()
    audit=pd.read_parquet(O/'Chu_C23_exposure_audit.parquet')
    extra=['source_excel_row','either_endpoint_screen_match','available_gene_symbol_screen_overlap']
    pairs=pairs.merge(audit[extra],on='source_excel_row',validate='one_to_one')
    assert len(pairs)==1743
    for endpoint,column in [('wt','wt_sequence'),('rt','variant_sequence')]:
        pairs['DEGRONOPEDIA_C23__'+endpoint]=score.loc[pairs[column].str[-23:]].to_numpy()
    predictions={'DEGRONOPEDIA_C23':(pairs.DEGRONOPEDIA_C23__wt.to_numpy(),pairs.DEGRONOPEDIA_C23__rt.to_numpy())}
    for key in ['GFP_MP__A','GFP_MP__T','MP_mCherry__A','MP_mCherry__T','QCDPred_normalized_cargo',
                'PAP_full_cargo','PAP_full_cargo__source_WT_calibrated']:
        predictions[key]=(pairs[key+'__wt'].to_numpy(),pairs[key+'__rt'].to_numpy())
    neural=pd.read_parquet(P/'pap_neural_comparison/frozen_predictions.parquet').set_index('sequence')
    for arch in ['cnn2w1','cnn2w7']:
        for suffix in ['', '__source_WT_calibrated']:
            key=arch+'__full_cargo__full'+suffix
            predictions[key]=(neural.loc[pairs.wt_sequence,key].to_numpy(),neural.loc[pairs.variant_sequence,key].to_numpy())
    masks={
        'all1743_provenance_limited':np.ones(len(pairs),dtype=bool),
        'either_endpoint_screen_matched':pairs.either_endpoint_screen_match,
        'neither_endpoint_screen_matched':~pairs.either_endpoint_screen_match,
        'neither_endpoint_or_available_symbol_overlap':~pairs.either_endpoint_screen_match&~pairs.available_gene_symbol_screen_overlap}
    expected=[1743,1454,289,97]
    metrics=[];contrasts=[]
    output=pairs[['source_excel_row','target_dependence_group','wt_sequence','variant_sequence','wt_mean','rt_mean',
        'observed_delta','observed_delta_r1','observed_delta_r2','either_endpoint_screen_match','available_gene_symbol_screen_overlap']].copy()
    for key,(wt,rt) in predictions.items():
        output[key+'__wt']=wt;output[key+'__rt']=rt;output[key+'__delta']=rt-wt
    for i,(label,mask) in enumerate(masks.items()):
        assert int(np.sum(mask))==expected[i]
        output[label+'_eligible']=np.asarray(mask)
        for key,(wt,rt) in predictions.items():
            delta=rt-wt
            metrics.append(dict(cohort=label,model=key,n=int(np.sum(mask)),
                groups=int(pairs.loc[mask,'target_dependence_group'].nunique()),
                native_WT_rho=rho(pairs.loc[mask,'wt_mean'],wt[mask]),native_RT_rho=rho(pairs.loc[mask,'rt_mean'],rt[mask]),
                delta_rho=rho(pairs.loc[mask,'observed_delta'],delta[mask]),
                replicate1_delta_rho=rho(pairs.loc[mask,'observed_delta_r1'],delta[mask]),
                replicate2_delta_rho=rho(pairs.loc[mask,'observed_delta_r2'],delta[mask]),
                predicted_zero_fraction=float(np.mean(abs(delta[mask])<1e-14))))
        if label=='either_endpoint_screen_matched':continue
        comparisons=[('DEGRONOPEDIA_C23','cnn2w1__full_cargo__full','C-model minus raw CNN1; conditional'),
            ('DEGRONOPEDIA_C23','PAP_full_cargo__source_WT_calibrated','C-model minus calibrated linear PAP; conditional')]
        pp={k:output.loc[mask,k+'__delta'].to_numpy() for row in comparisons for k in row[:2]}
        cc=bootstrap(pairs.loc[mask,'observed_delta'].to_numpy(),pairs.loc[mask,'target_dependence_group'].to_numpy(),pp,comparisons,20261108+i)
        for row in cc:row['cohort']=label
        contrasts.extend(cc)
    output.to_parquet(O/'evaluated_predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(O/'transfer_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(O/'paired_contrasts.csv',index=False)
    receipt=dict(status='evaluated; independent source/metric verification pending',target_fitting=False,
        prediction_hash_unchanged=True,models=len(predictions),metric_rows=len(metrics),all_exposure_strata_retained=True,
        actual90_10_training_split_unavailable=True,available_symbol_sensitivity_is_not_certified_gene_independence=True,
        intervals='conditional operational-group bootstrap; no retraining uncertainty or multiplicity-adjusted discovery')
    (O/'EVALUATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(metrics).query("model=='DEGRONOPEDIA_C23' or model=='cnn2w1__full_cargo__full'").to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)

if __name__=='__main__':main()
