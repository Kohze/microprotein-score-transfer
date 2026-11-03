"""Report all raw C23/C30/full58 PAP windows in the terminal challenge."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from evaluate_pap_neural_models import rho,bootstrap

P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'

def main():
    freeze=json.loads((O/'WINDOW_CONTROL_FREEZE.json').read_text())
    assert hashlib.sha256((O/'frozen_window_controls.parquet').read_bytes()).hexdigest()==freeze['prediction_sha256']
    pairs=pd.read_parquet(O/'evaluated_predictions.parquet')
    small=pd.read_parquet(O/'frozen_window_controls.parquet').set_index('C23_sequence')
    prior=pd.read_parquet(P/'pap_neural_comparison/frozen_predictions.parquet').set_index('sequence')
    predictions={}
    for arch in ['cnn2w1','cnn2w7']:
        key=arch+'__C23__full';predictions[key]=(small.loc[pairs.wt_sequence.str[-23:],key].to_numpy(),small.loc[pairs.variant_sequence.str[-23:],key].to_numpy())
        key=arch+'__C30__full';predictions[key]=(prior.loc[pairs.wt_sequence,key].to_numpy(),prior.loc[pairs.variant_sequence,key].to_numpy())
    masks={c.removesuffix('_eligible'):pairs[c].to_numpy() for c in pairs if c.endswith('_eligible')}
    metrics=[];contrasts=[]
    for key,(wt,rt) in predictions.items():
        pairs[key+'__wt']=wt;pairs[key+'__rt']=rt;pairs[key+'__delta']=rt-wt
    for i,(label,mask) in enumerate(masks.items()):
        for key,(wt,rt) in predictions.items():
            delta=rt-wt
            metrics.append(dict(cohort=label,model=key,n=int(np.sum(mask)),groups=int(pairs.loc[mask,'target_dependence_group'].nunique()),
                native_WT_rho=rho(pairs.loc[mask,'wt_mean'],wt[mask]),native_RT_rho=rho(pairs.loc[mask,'rt_mean'],rt[mask]),
                delta_rho=rho(pairs.loc[mask,'observed_delta'],delta[mask]),
                replicate1_delta_rho=rho(pairs.loc[mask,'observed_delta_r1'],delta[mask]),replicate2_delta_rho=rho(pairs.loc[mask,'observed_delta_r2'],delta[mask]),
                predicted_zero_fraction=float(np.mean(abs(delta[mask])<1e-14))))
        if label=='either_endpoint_screen_matched':continue
        comparisons=[('DEGRONOPEDIA_C23','cnn2w1__C23__full','post-hoc same C23 input; training sources differ'),
                     ('DEGRONOPEDIA_C23','cnn2w1__C30__full','post-hoc window-context C30 reference')]
        values={k:pairs.loc[mask,k+'__delta'].to_numpy() for row in comparisons for k in row[:2]}
        rows=bootstrap(pairs.loc[mask,'observed_delta'].to_numpy(),pairs.loc[mask,'target_dependence_group'].to_numpy(),values,comparisons,20261112+i)
        for row in rows:row['cohort']=label
        contrasts.extend(rows)
    pairs.to_parquet(O/'evaluated_window_predictions.parquet',index=False)
    pd.DataFrame(metrics).to_csv(O/'window_control_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(O/'window_control_contrasts.csv',index=False)
    print(pd.DataFrame(metrics).query("cohort=='all1743_provenance_limited' or cohort=='neither_endpoint_screen_matched'").to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)

if __name__=='__main__':main()
