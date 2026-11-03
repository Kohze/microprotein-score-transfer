"""Freeze published neural native/shuffle scores and source-only calibration."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
B=P.parent/'stability_benchmark_2026-10-31'
O=P/'pap_neural_comparison'
sys.path.insert(0,str(B))
import run_benchmark as b
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from pap_neural_runtime import PublishedPAP,keras,torch


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def shuffled(s,seed,n=19):
    if len(s)<=10:
        return [s]*n
    start=int.from_bytes(hashlib.sha256(f'{seed}|{s}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(start)
    middle=np.array(list(s[5:-5]))
    return [s[:5]+''.join(rng.permutation(middle))+s[-5:] for _ in range(n)]


def main():
    started=time.time()
    native=P/'independent_data/Voutsinos_author_holdout_sequence_audit.parquet'
    paired=P/'independent_interventions/sequence_pair_audit.parquet'
    v=pd.read_parquet(native,columns=['aa'])
    c=pd.read_parquet(paired,columns=['wt_sequence','variant_sequence'])
    seqs=sorted(set(b.A.protein_sequence)|set(v.aa)|set(c.wt_sequence)|set(c.variant_sequence))
    assert min(map(len,seqs))>=7
    input_seqs=sorted(set(seqs)|{s[-30:] for s in seqs},key=lambda s:(len(s),s))
    bylength={n:[s for s in input_seqs if len(s)==n] for n in sorted(set(map(len,input_seqs)))}
    results=pd.DataFrame(index=input_seqs)
    checks=[]
    for name in ['cnn2w1','cnn2w7']:
        model=PublishedPAP(name)
        # Samples are fixed from sequences/lengths, before target associations.
        for length,group in bylength.items():
            step=max(1,len(group)//8)
            sample=group[::step][:8]
            expected=model.numpy_predict(sample)
            actual=model.predict(sample)
            err=float(np.max(abs(expected-actual)))
            assert err<1e-5,(name,length,err)
            checks.append(dict(model=name,length=length,n=len(sample),NumPy_max_error=err))
        for length,group in bylength.items():
            actual=model.predict(group)
            for j,component in enumerate(['full','internal','terminal']):
                results.loc[group,name+'__'+component]=actual[:,j]
            for seed in [20261105,20261106]:
                label='shuffle1' if seed==20261105 else 'shuffle2'
                if name=='cnn2w1':
                    # Mathematical composition/last5 invariance, explicitly
                    # checked on fixed samples; not a claim of19N evaluations.
                    sample=group[:min(4,len(group))]
                    shuffled_sample=[z for s in sample for z in shuffled(s,seed)]
                    predictions=model.predict(shuffled_sample)
                    reference=np.repeat(model.predict(sample),19,axis=0)
                    err=float(np.max(abs(predictions-reference)))
                    assert err<1e-5
                    results.loc[group,name+'__'+label+'_mean']=actual[:,0]
                    results.loc[group,name+'__'+label+'_sd']=0.
                else:
                    for start in range(0,len(group),128):
                        batch=group[start:start+128]
                        shuf=[z for s in batch for z in shuffled(s,seed)]
                        internal=model.internal(shuf).reshape(len(batch),19)
                        # First/final5 are fixed, so the published terminal
                        # branch is identical across all19 realizations.
                        ct=results.loc[batch,name+'__terminal'].to_numpy()
                        score=internal+ct[:,None]
                        results.loc[batch,name+'__'+label+'_mean']=score.mean(axis=1)
                        results.loc[batch,name+'__'+label+'_sd']=score.std(axis=1,ddof=1)
            progress=dict(model=name,length=length,sequences=len(group),elapsed_seconds=round(time.time()-started,1))
            (O/'FREEZE_PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress),flush=True)
    results.index.name='input_sequence'
    results.to_parquet(O/'frozen_input_scores.parquet')
    pred=pd.DataFrame(dict(sequence=seqs))
    source=b.A[b.A.source_sheet.eq('GFP_MP')&b.A.group.eq('microprotein_WT')&b.A.mean_stability.notna()&~b.A.inconsistent_parent_id]
    maps={}
    for name in ['cnn2w1','cnn2w7']:
        for representation in ['C30','full_cargo']:
            targets=[s[-30:] if representation=='C30' else s for s in seqs]
            train=[s[-30:] if representation=='C30' else s for s in source.protein_sequence]
            for component in ['full','internal','terminal','shuffle1_mean','shuffle2_mean']:
                key=name+'__'+representation+'__'+component
                rawkey=name+'__'+component
                pred[key]=results.loc[targets,rawkey].to_numpy()
                x=results.loc[train,rawkey].to_numpy()
                iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(x,source.mean_stability)
                pred[key+'__source_WT_calibrated']=iso.predict(pred[key])
                maps[key]=dict(source='Guo original GFP_MP WT only',n_source=len(source),increasing=True,
                    x_thresholds=iso.X_thresholds_.tolist(),y_thresholds=iso.y_thresholds_.tolist(),
                    source_min=float(x.min()),source_max=float(x.max()),
                    target_input_sequences_outside_source_range=int(((pred[key]<x.min())|(pred[key]>x.max())).sum()),
                    target_outcomes_used=False)
    file=O/'frozen_predictions.parquet'
    pred.to_parquet(file,index=False)
    (O/'SOURCE_ONLY_CALIBRATIONS.json').write_text(json.dumps(maps,indent=2)+'\n')
    (O/'INFERENCE_IMPLEMENTATION_CHECK.json').write_text(json.dumps(dict(status='passed',checks=checks,
        NumPy_max_absolute_error=max(r['NumPy_max_error'] for r in checks),
        all_named_HDF5_tensors_checked=True,cnn2w1_order_invariance_sampled=True,
        backend_adaptation='backend import and explicit legacy numeric layer loading only; published graph preserved'),indent=2)+'\n')
    runtime=dict(python=sys.version,keras=keras.__version__,torch=torch.__version__,backend=keras.backend.backend(),
        packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','h5py','scipy','scikit-learn']})
    (O/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    freeze=dict(status='published neural native/shuffle predictions frozen before new neural associations',
        prediction_sha256=sha(file),protocol_sha256=sha(O/'COMPARISON_PROTOCOL.txt'),
        source_code_sha256=sha(Path(__file__)),runtime_adapter_sha256=sha(P/'pap_neural_runtime.py'),
        published_source_receipt_sha256=sha(O/'SOURCE_RECEIPT.json'),
        native_sequence_membership_sha256=sha(native),paired_sequence_membership_sha256=sha(paired),
        source_sequences=len(pred),unique_network_inputs=len(results),target_model_fitting=False,
        calibration_source='Guo GFP_MP WT only',other_predictor_target_results_already_known=True,
        shuffles_per_seed=19,seeds=[20261105,20261106],cnn2w1_full_shuffle_mean='analytic identity with fixed-sample numerical checks',
        cnn2w7_shuffles='explicitly evaluated',Voutsinos_training_provenance='author-held-out within-study reference',
        elapsed_seconds=round(time.time()-started,1))
    (O/'PREDICTION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze,indent=2),flush=True)


if __name__=='__main__':
    with b.threadpool_limits(limits=4):
        main()
