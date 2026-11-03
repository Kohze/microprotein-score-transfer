"""Freeze original C-terminal PSI predictions before terminal-model associations."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'
sys.path.insert(0,str(P.parent/'stability_benchmark_2026-10-31'))
import run_benchmark as b
sys.path.insert(0,str(O/'vendor'))
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import runpy
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    start=time.perf_counter()
    protocol=O/'COMPARISON_PROTOCOL.txt'
    protocol.write_text('''DEGRONOPEDIA C-TERMINAL TRANSFER -SECONDARY EXPOSURE-STRATIFIED CHALLENGE
Other predictor associations and the development-screen exposure counts have
already been inspected; no terminal model versus target associations have been
computed. This is retrospective, not preregistered or blinded research.

Use only the frozen published C-terminal CatBoost weights at pinned commit
2703570145c3c53cf276b705e1a6cc59083dfd12 and the original descriptor/inference
functions. All2997 named features are supplied in the original order. Do not
apply the unused columnSet regex. Current runtime reproduced all12 released
two-decimal example predictions. Exact original dependency versions are not
pinned by the published conda file, so cross-version parity is limited to those
examples. No retraining, target calibration, sign inversion or favorable model
choice. Higher predicted PSI means greater terminal reporter stability.

Inputs are the final23 residues of both endpoints of the existing1743 Chu
common pairs, all standard58mers. This exactly matches the C input/descriptor
rule; both endpoints remain58mer window replacements, not measured-WT appends.
Preserve all existing primary pair memberships and gene/identical-cargo groups.
No N-model predictions are made for this C-terminal exposed target. N-model
weights/examples remain audited, but native initiator processing is not assumed.

Report paired delta (RT-minus-WT), native WT and RT rank correlations and both
replicate delta correlations for: all1743 pairs (provenance-limited), either
endpoint screen-matched1454, neither-endpoint screen-matched289, and neither
endpoint nor available gene-symbol overlap97. The latter is a conservative
available-symbol sensitivity, not an alias-complete gene-disjoint certificate.
An exact match proves development-screen membership, not actual membership of
the unavailable90/10 training split. Do not rank all1743 as an equally
independent competitor to provenance-filtered PAP/Guo heads.

Retain frozen comparators on every same cohort: both source A/T contexts,
normalized QCDPred, linear full58 PAP raw/source-WT-calibrated, and neural
CNN1/CNN7 full58 raw/source-WT-calibrated. No cross-scale raw MAE comparison.
Use1000 paired operational-group bootstrap resamples for C-model-minus-raw
CNN1 and C-model-minus-calibrated-linear delta-rho contrasts for all1743,
the289 endpoint-unmatched cohort and97 available-symbol sensitivity. These
are descriptive conditional intervals, not full retraining uncertainty,
multiplicity-adjusted discovery, proof of an exposure cause, or native nORF
function. Both seen- and unseen-anchor results must be retained.

Save model/source/input hashes and frozen output before evaluating new model
associations. Retain all results, including adverse transfer and wide intervals.
''',encoding='utf-8')
    pairs=pd.read_parquet(O/'Chu_C23_exposure_audit.parquet')
    pairs=pairs[pairs.primary60_common_eligible]
    assert len(pairs)==1743
    sequences=sorted(set(pairs.WT_C23)|set(pairs.RT_C23))
    assert all(len(s)==23 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') for s in sequences)
    ns=runpy.run_path(str(O/'published/calculate-desc.py'),run_name='original_descriptor_functions')
    model=CatBoostRegressor();model.load_model(str(O/'published/models/c-terminus.cbm'))
    names=model.feature_names_
    def descriptors(s):return ns['generate_ML_tsv'](s,ns['generate_ogryzki'](s)[0])[names]
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i in range(0,len(sequences),128):
            seq=sequences[i:i+128]
            features=pd.concat(list(pool.map(descriptors,seq)),ignore_index=True)
            values=model.predict(features,thread_count=4)
            assert np.isfinite(values).all()
            rows.extend(dict(C23_sequence=s,DEGRONOPEDIA_C23=float(v)) for s,v in zip(seq,values))
            write(O/'FREEZE_PROGRESS.json',dict(completed=len(rows),total=len(sequences),elapsed_seconds=round(time.perf_counter()-start,2)))
            print('frozen C23 scores',len(rows),'/',len(sequences),'seconds',round(time.perf_counter()-start,1),flush=True)
    predictions=pd.DataFrame(rows);output=O/'frozen_predictions.parquet'
    predictions.to_parquet(output,index=False)
    predictions.to_csv(O/'frozen_predictions.csv',index=False)
    write(O/'PREDICTION_FREEZE.json',dict(status='C-model predictions frozen before its new target associations',
        prediction_sha256=sha(output),protocol_sha256=sha(protocol),source_code_sha256=sha(Path(__file__)),
        model_sha256=sha(O/'published/models/c-terminus.cbm'),published_source_sha256=sha(O/'published/calculate-desc.py'),
        pair_membership_sha256=sha(O/'Chu_C23_exposure_audit.parquet'),unique_C23_inputs=len(sequences),pairs=1743,
        source_model_retraining=False,target_outcomes_used_for_fitting=False,other_model_results_already_known=True,
        features=2997,elapsed_seconds=round(time.perf_counter()-start,2)))

if __name__=='__main__':
    with b.threadpool_limits(limits=4):main()
