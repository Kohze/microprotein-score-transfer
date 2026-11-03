"""Post-hoc evaluation restriction prompted by newly accessible STAR Methods."""
from pathlib import Path
import hashlib
import json
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
WS = P.parent
from scipy.stats import spearmanr

O = P / 'native_function_bridge'
INPUT = WS / 'stability_benchmark_2026-10-31/results/mutation_predictions.parquet'


def rho(y, p):
    return float(spearmanr(y, p).statistic) if np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan


def main():
    (O / 'GUO_METHODS_SENSITIVITY_PROTOCOL.txt').write_text('''POST-HOC SOURCE TERMINAL-LEUCINE FILTER SENSITIVITY
Newly accessible Guo STAR Methods exclude microproteins with leucine at -1/-2
from the global Leu-to-Ala analysis. The paper's Figure 4 caption also states
this exclusion for WT/Leu-to-Ala/extension comparison. Our existing benchmark
retained all sequence-valid measured pairs. Earlier results are known.

Retain every original prediction and fit. Evaluate all original predictor
variants on all, source-rule-retained and source-rule-omitted pairs within
each edit family. The restriction concerns test-population interpretation,
not a new fit or reproduction of all original assay-processing decisions.
Retain mean and both replicate effects, raw rank and compatible-scale errors.
Use 1,000 operational-group resamples for fixed T-minus-A, O-minus-T and
calibrated-minus-raw full-PAP rank contrasts. No multiplicity-adjusted discovery,
causal assignment, target fitting or sign choice. Preserve all prior results.
''', encoding='utf-8')
    before = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    data = pd.read_parquet(INPUT)
    data['source_terminal_Leu_filter_pass'] = ~data.protein_sequence_WT.str[-2:].str.contains('L')
    predictors = ['zero','family_mean','cross_family_mean','L','A','T','O','Shuffle0',
        'AAC_resemblance','AAC_resemblance_calibrated','PAP_Cterminal','PAP_Cterminal_calibrated',
        'PAP_full','PAP_full_calibrated','delta_trained_A','delta_trained_T','delta_trained_O',
        'cross_family_A','cross_family_T','cross_family_O']
    metrics = []
    contrasts = []
    cohorts = []
    for family, f in data.groupby('mutation_group', sort=True):
        for stratum, q in [('all',f),('source_rule_retained',f[f.source_terminal_Leu_filter_pass]),
                           ('source_rule_omitted',f[~f.source_terminal_Leu_filter_pass])]:
            q = q.reset_index(drop=True)
            cohorts.append(dict(family=family,stratum=stratum,n=len(q),groups=q.eval_group.nunique()))
            y = q.delta.to_numpy()
            for outcome, target in [('mean',y),('replicate1',(q.stability_r1_mut-q.stability_r1_WT).to_numpy()),
                                     ('replicate2',(q.stability_r2_mut-q.stability_r2_WT).to_numpy())]:
                for m in predictors:
                    p = q[m].to_numpy()
                    raw_incompatible = m in ['AAC_resemblance','PAP_Cterminal','PAP_full']
                    metrics.append(dict(family=family,stratum=stratum,outcome=outcome,model=m,
                        n=len(q),groups=q.eval_group.nunique(),rho=rho(target,p),
                        MAE=np.nan if raw_incompatible else float(np.mean(abs(target-p)))))
            # Index groups once; resample groups with replacement, preserving all rows.
            group_indices = [np.flatnonzero(q.eval_group.to_numpy() == g) for g in sorted(q.eval_group.unique())]
            rng = np.random.default_rng(20261101)
            specs = [('T','A'),('O','T'),('PAP_full_calibrated','PAP_full')]
            values = {a+' minus '+b:[] for a,b in specs}
            for _ in range(1000):
                ix = np.concatenate([group_indices[i] for i in rng.integers(0,len(group_indices),len(group_indices))])
                for a,b in specs:
                    values[a+' minus '+b].append(rho(y[ix],q[a].to_numpy()[ix])-rho(y[ix],q[b].to_numpy()[ix]))
            for a,b in specs:
                vals = np.asarray(values[a+' minus '+b])
                lo,hi = np.nanquantile(vals,[.025,.975])
                contrasts.append(dict(family=family,stratum=stratum,contrast=a+' minus '+b,n=len(q),
                    groups=len(group_indices),estimate=rho(y,q[a].to_numpy())-rho(y,q[b].to_numpy()),
                    low=lo,high=hi,bootstrap_replicates=1000,
                    scope='post-hoc evaluation restriction; conditional on original fits; no multiplicity adjustment'))
            print(family,stratum,len(q),'finished',flush=True)
    pd.DataFrame(metrics).to_csv(O / 'source_terminal_filter_metrics.csv',index=False)
    pd.DataFrame(contrasts).to_csv(O / 'source_terminal_filter_contrasts.csv',index=False)
    pd.DataFrame(cohorts).to_csv(O / 'source_terminal_filter_cohorts.csv',index=False)
    data[['protein_identifier','protein_sequence_WT','mutation_group','eval_group','source_terminal_Leu_filter_pass']].to_parquet(
        O / 'source_terminal_filter_membership.parquet',index=False)
    assert before == hashlib.sha256(INPUT.read_bytes()).hexdigest()
    (O / 'GUO_TERMINAL_FILTER_RECEIPT.json').write_text(json.dumps(dict(
        status='calculated; independent point/coverage verification pending',
        source_prediction_sha256=before, existing_predictions_and_fits_unchanged=True,
        metric_rows=len(metrics), contrast_rows=len(contrasts), cohorts=cohorts,
        source_paper_filter_is_not_our_prior_benchmark_filter=True,
        new_target_fitting=False, native_function_inference=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
