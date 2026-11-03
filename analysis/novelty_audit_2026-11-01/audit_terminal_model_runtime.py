"""Inspect pinned CatBoost metadata and reproduce original terminal examples."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'
sys.path.insert(0,str(P.parent/'stability_benchmark_2026-10-31'))
import run_benchmark as b
sys.path.insert(0,str(O/'vendor'))
import hashlib
import importlib.metadata
import json
import re
import runpy
import numpy as np
from catboost import CatBoostRegressor
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def main():
    source=O/'published/calculate-desc.py'
    namespace=runpy.run_path(str(source),run_name='published_descriptor_functions')
    expected_text=(O/'published/tests/results-ok.txt').read_text()
    expected=[float(x) for x in re.findall(r'Predicted PSI: ([0-9.]+)',expected_text)]
    models={};metadata={}
    for key,name in [('C','c-terminus.cbm'),('NiMetYes','n-terminus_iMetYES.cbm'),('NiMetNo','n-terminus_iMetNO.cbm')]:
        model=CatBoostRegressor();model.load_model(str(O/'published/models'/name))
        models[key]=model
        raw={k:v for k,v in model.get_metadata().items()}
        parsed={k:json.loads(v) if v.startswith(('{','[')) else v for k,v in raw.items()}
        metadata[key]=dict(features=model.feature_names_,categorical_indices=model.get_cat_feature_indices(),
            trees=model.tree_count_,parameters=model.get_all_params(),metadata=parsed,
            weight_sha256=hashlib.sha256((O/'published/models'/name).read_bytes()).hexdigest())
    (O/'MODEL_METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
    observed=[];examples=[]
    for number in range(4):
        sequence=(O/'published/seqs'/f'{number}.txt').read_text().strip()
        fragments=namespace['generate_ogryzki'](sequence)
        for kind in ['C','NiMetYes','NiMetNo']:
            window=sequence[-23:] if kind=='C' else sequence[1:24] if sequence.startswith('M') else sequence[:23]
            original=namespace['generate_ML_tsv'](window,fragments[0 if kind=='C' else 1])
            model=models[kind];features=original[model.feature_names_]
            value=float(model.predict(features,thread_count=4)[0]);observed.append(value)
            examples.append(dict(example=number,terminus=kind,prediction=value,published_rounded_prediction=expected[len(observed)-1],
                feature_count=len(model.feature_names_),input_window=window))
            features.to_parquet(O/f'example_{number}_{kind}_descriptors.parquet')
    assert len(expected)==len(observed)==12
    assert all(abs(a-e)<=.005001 for a,e in zip(observed,expected))
    receipt=dict(status='passed original12 rounded-example predictions',examples=examples,
        versions={k:importlib.metadata.version(k) for k in ['catboost','peptides','rdkit','numpy','pandas']},
        metadata_file='MODEL_METADATA.json',original_feature_and_inference_functions_used=True,
        target_predictions_evaluated=False,training_split_sequence_membership_available=False,
        processing='Both N models exclude initiator M from descriptors; learned weights differ. Do not assume native cleavage state.')
    (O/'RUNTIME_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],features={k:len(v.feature_names_) for k,v in models.items()},
        categorical={k:len(v.get_cat_feature_indices()) for k,v in models.items()}),indent=2),flush=True)

if __name__=='__main__':
    with b.threadpool_limits(limits=4):main()
