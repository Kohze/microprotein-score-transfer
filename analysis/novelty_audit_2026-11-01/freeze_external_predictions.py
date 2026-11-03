"""Fit source-only models and freeze predictions before new associations."""
from pathlib import Path
import os
import sys
P = Path(__file__).resolve().parent
B = P.parent / 'stability_benchmark_2026-10-31'
sys.path.insert(0, str(B))
import run_benchmark as b
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = P / 'independent_test'
OUT.mkdir(exist_ok=True)
AA = 'ACDEFGHIKLMNPQRSTVWY'
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(seq, param):
    size = len(seq)
    f = dict(sequence=seq, log_length=np.log(size), log_length_sq=np.log(size)**2)
    f.update({'AAC_' + a: seq.count(a)/size for a in AA})
    for j in range(5):
        f.update({f'N{j+1}_{a}': float(seq[j] == a) for a in AA})
        f.update({f'C{j+1}_{a}': float(seq[-j-1] == a) for a in AA})
    interior = seq[size//2-2:size//2+2]
    assert len(interior) == 4
    for j, residue in enumerate(interior):
        f.update({f'I{j+1}_{a}': float(residue == a) for a in AA})
    for name, s in [('hydropathy', seq), ('N10_hydropathy', seq[:10]), ('C10_hydropathy', seq[-10:])]:
        f[name] = float(np.mean([HYDRO[a] for a in s]))
    comp = sum(param['comp'][a] for a in seq) * 30 / size
    term = 0.
    for family in ['pssm', 'pair', 'triplet']:
        for motif, coefficient in param[family].items():
            if all(size >= int(t[1:]) and seq[-int(t[1:])] == t[0] for t in motif.split(':')):
                term += coefficient
    f.update(PAP_composition=float(comp), PAP_terminal=float(term), PAP_full=float(param['intercept'] + comp + term))
    return f


def main():
    protocol = P / 'INDEPENDENT_COMPARISON_PLAN.txt'
    target = pd.read_parquet(P / 'independent_data/Voutsinos_author_holdout_sequence_audit.parquet')
    # Only the previously saved sequence/membership audit is read here.
    assert not any(c in target for c in ['abundance_score', 'degron_score'])
    paramfile = B / 'inputs/PAP_HUMAN30_PARSED_COEFFICIENTS.json'
    param = json.loads(paramfile.read_text())
    src = b.A[b.A.group.eq('microprotein_WT') & b.A.source_sheet.isin(['GFP_MP', 'MP_mCherry']) &
              b.A.mean_stability.notna() & ~b.A.inconsistent_parent_id].copy()
    sequences = sorted(set(src.protein_sequence) | set(target.aa))
    f = pd.DataFrame([features(s, param) for s in sequences]).set_index('sequence')
    common = sorted(set(src.protein_sequence))
    old = b.F.loc[common, b.SETS['T']].to_numpy(float)
    new = f.loc[common, b.SETS['T']].to_numpy(float)
    difference = float(np.max(abs(old-new)))
    assert difference < 1e-12, difference
    specs = {k: b.SETS[k] for k in ['A', 'T']}
    specs['T_noPAP'] = [c for c in b.SETS['T'] if not c.startswith('PAP_')]
    specs['N4'] = b.SETS['A'] + [f'N{j}_{a}' for j in range(2, 6) for a in AA]
    specs['C4'] = b.SETS['A'] + [f'C{j}_{a}' for j in range(1, 5) for a in AA]
    specs['I4'] = b.SETS['A'] + [f'I{j}_{a}' for j in range(1, 5) for a in AA]
    # N4 omits the usually constant initiator methionine position; C4 and I4
    # use the same 80 one-hot columns. No diagnostic is selected on target data.
    (OUT / 'FEATURE_SPECIFICATION.json').write_text(json.dumps(dict(
        models=specs, diagnostics='secondary; N4/C4/I4 have 80 added one-hot columns each',
        interior_definition='four central residues, zero-based slice L//2-2:L//2+2',
        source_feature_max_difference=difference, all_target_outcomes_used=False), indent=2) + '\n')
    f.to_parquet(OUT / 'features.parquet')
    receipts, source_predictions = [], []
    predictions = target[['name', 'aa', 'protein_id']].copy()
    predictions['PAP_internal'] = f.loc[target.aa, 'PAP_composition'].to_numpy()
    predictions['PAP_terminal'] = f.loc[target.aa, 'PAP_terminal'].to_numpy()
    predictions['PAP_full'] = f.loc[target.aa, 'PAP_full'].to_numpy()
    for source in ['GFP_MP', 'MP_mCherry']:
        d = src[src.source_sheet.eq(source)].reset_index(drop=True)
        groups = b.groups(d.protein_sequence)
        y = d.mean_stability.to_numpy()
        ff = b.folds(groups)
        for model, cols in specs.items():
            print(source, model, flush=True)
            x = f.loc[d.protein_sequence, cols].to_numpy(float)
            zz = f.loc[target.aa, cols].to_numpy(float)
            pp, alpha, tuning = b.fit_predict(x, y, groups, zz, b.SEED)
            predictions[source + '__' + model] = pp
            receipts.append(dict(source=source, model=model, purpose='external prediction', alpha=alpha,
                inner_mae=tuning, source_rows=len(d), source_groups=int(len(np.unique(groups))),
                nominal_features=len(cols), variable_features=int((x.std(0)>1e-12).sum()), target_outcomes_used=False))
            oof = np.full(len(d), np.nan)
            for k in range(5):
                tr, te = ff != k, ff == k
                assert not set(groups[tr]) & set(groups[te])
                oof[te], chosen, _ = b.fit_predict(x[tr], y[tr], groups[tr], x[te], b.SEED+k+19)
                receipts.append(dict(source=source, model=model, purpose='source nested CV', fold=k,
                    alpha=chosen, train_rows=int(tr.sum()), test_rows=int(te.sum()), group_overlap=0))
            source_predictions.append(pd.DataFrame(dict(source=source, model=model,
                sequence=d.protein_sequence, group=groups, fold=ff, observed=y, predicted=oof)))
    output = OUT / 'frozen_target_predictions.parquet'
    predictions.to_parquet(output, index=False)
    source_oof = pd.concat(source_predictions, ignore_index=True)
    source_oof.to_parquet(OUT / 'source_nested_predictions.parquet', index=False)
    source_metrics = []
    for (source, model), z in source_oof.groupby(['source', 'model']):
        source_metrics.append(dict(source=source, model=model, n=len(z),
            rho=float(spearmanr(z.observed,z.predicted).statistic), mae=float(abs(z.observed-z.predicted).mean())))
    pd.DataFrame(source_metrics).to_csv(OUT / 'source_nested_metrics.csv', index=False)
    (OUT / 'FIT_RECEIPTS.json').write_text(json.dumps(receipts, indent=2) + '\n')
    freeze = dict(status='source-only predictions frozen; new target associations not evaluated',
        protocol_sha256=sha(protocol), prediction_sha256=sha(output), source_code_sha256=sha(Path(__file__)),
        source_feature_equivalence_max_error=difference, target_rows=len(predictions),
        raw_target_outcome_file_loaded_by_this_script=False, evaluation_direction='higher predicts greater abundance; no target-dependent reversal',
        feature_specification_sha256=sha(OUT / 'FEATURE_SPECIFICATION.json'),
        source_features_sha256=sha(B/'sequence_features.parquet'),
        source_groups_sha256=sha(B/'sequence_dependence_groups.parquet'),
        pretrained_PAP_coefficients_sha256=sha(paramfile))
    (OUT / 'PREDICTION_FREEZE.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print(json.dumps(freeze, indent=2), flush=True)


if __name__ == '__main__':
    with b.threadpool_limits(limits=4):
        main()
