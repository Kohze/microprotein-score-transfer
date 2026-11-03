"""Freeze independent C-terminal predictions and source-WT-only calibration."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

P=Path(__file__).resolve().parent
B=P.parent/'stability_benchmark_2026-10-31'
O=P/'independent_interventions'
sys.path.insert(0,str(B))
import run_benchmark as b
from sklearn.isotonic import IsotonicRegression
from freeze_external_predictions import features
from freeze_published_baselines import literal_model


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main():
    audit=O/'sequence_pair_audit.parquet'
    pairs=pd.read_parquet(audit)
    paramfile=B/'inputs/PAP_HUMAN30_PARSED_COEFFICIENTS.json'
    param=json.loads(paramfile.read_text())
    src=b.A[b.A.group.eq('microprotein_WT')&b.A.source_sheet.isin(['GFP_MP','MP_mCherry'])&
            b.A.mean_stability.notna()&~b.A.inconsistent_parent_id].copy()
    seqs=sorted(set(pairs.wt_sequence)|set(pairs.variant_sequence)|set(src.protein_sequence))
    ff=pd.DataFrame([features(s,param) for s in seqs]).set_index('sequence')
    common=sorted(set(src.protein_sequence))
    error=float(np.max(abs(ff.loc[common,b.SETS['T']].to_numpy()-b.F.loc[common,b.SETS['T']].to_numpy())))
    assert error<1e-12
    ff.to_parquet(O/'features.parquet')
    target=sorted(set(pairs.wt_sequence)|set(pairs.variant_sequence))
    pred=pd.DataFrame(dict(sequence=target))
    receipts=[]
    prior=json.loads((P/'independent_test/FIT_RECEIPTS.json').read_text())
    for context in ['GFP_MP','MP_mCherry']:
        d=src[src.source_sheet.eq(context)]
        groups=b.groups(d.protein_sequence)
        for model in ['A','T']:
            x=ff.loc[d.protein_sequence,b.SETS[model]].to_numpy(float)
            z=ff.loc[target,b.SETS[model]].to_numpy(float)
            values,alpha,tuning=b.fit_predict(x,d.mean_stability.to_numpy(),groups,z,b.SEED)
            old=next(r for r in prior if r['source']==context and r['model']==model and r.get('purpose')=='external prediction')
            assert alpha==old['alpha'] and np.max(abs(np.array(tuning)-old['inner_mae']))<1e-10
            pred[context+'__'+model]=values
            receipts.append(dict(source=context,model=model,alpha=alpha,inner_mae=tuning,
                source_rows=len(d),source_groups=int(len(np.unique(groups))),target_outcomes_used=False))
            print('source-only head',context,model,alpha,flush=True)
    qfile=P/'published_predictor_comparison/QCDpred.py'
    papfile=B/'inputs/Voutsinos2026_code/PAP.py'
    q=literal_model(qfile,'model')
    h=literal_model(papfile,'human25')
    def score(s):
        full=features(s,param)
        tail=features(s[-30:],param)
        windows=[s[i:i+17] for i in range(len(s)-16)]
        first=s[:25]
        return dict(PAP_C30=tail['PAP_full'],PAP_C30_composition=tail['PAP_composition'],
            PAP_full_cargo=full['PAP_full'],PAP_full_cargo_composition=full['PAP_composition'],
            PAP_terminal=full['PAP_terminal'],
            QCDPred_mean17=1-float(np.mean([expit(q['intersect']+sum(q[a] for a in w)) for w in windows])),
            QCDPred_normalized_cargo=1-float(expit(q['intersect']+sum(q[a] for a in s)*17/len(s))),
            human25_first25=1-float(expit(h['intercept']+sum(h['comp'][a] for a in first)*25/len(first))),
            human25_normalized_cargo=1-float(expit(h['intercept']+sum(h['comp'][a] for a in s)*25/len(s))))
    all_scores=pd.DataFrame([dict(sequence=s,**score(s)) for s in seqs]).set_index('sequence')
    source=src[src.source_sheet.eq('GFP_MP')]
    calibrations={}
    for model in all_scores:
        pred[model]=all_scores.loc[target,model].to_numpy()
        x=all_scores.loc[source.protein_sequence,model].to_numpy()
        y=source.mean_stability.to_numpy()
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,y)
        pred[model+'__source_WT_calibrated']=iso.predict(pred[model])
        pred[model+'__source_range_clipped']=(pred[model]<float(x.min()))|(pred[model]>float(x.max()))
        calibrations[model]=dict(source='original Guo GFP_MP WT only',increasing=True,
            source_rows=len(source),x_thresholds=iso.X_thresholds_.tolist(),y_thresholds=iso.y_thresholds_.tolist(),
            target_sequences_outside_source_range=int(pred[model+'__source_range_clipped'].sum()),
            target_outcomes_used=False)
    pred.to_parquet(O/'frozen_predictions.parquet',index=False)
    pred.to_csv(O/'frozen_predictions.csv',index=False)
    (O/'SOURCE_ONLY_FITS.json').write_text(json.dumps(receipts,indent=2)+'\n')
    (O/'SOURCE_ONLY_CALIBRATIONS.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    freeze=dict(status='predictions frozen before any Chu predictor/outcome associations',
        protocol_sha256=sha(O/'COMPARISON_PROTOCOL.txt'),sequence_audit_sha256=sha(audit),
        prediction_sha256=sha(O/'frozen_predictions.parquet'),source_code_sha256=sha(Path(__file__)),
        input_source_feature_max_difference=error,source_heads_match_prior_source_tuning=True,
        target_unique_sequences=len(pred),WT_calibration_source='Guo GFP_MP only; increasing isotonic',
        target_outcomes_read=False,target_outcomes_used_for_fitting=False,
        important_scope='58mer C-terminal sequence replacement reporters, not simple full-native-ORF extension')
    (O/'PREDICTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze,indent=2),flush=True)


if __name__=='__main__':
    with b.threadpool_limits(limits=4):
        main()
