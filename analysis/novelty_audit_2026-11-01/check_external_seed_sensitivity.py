"""Prespecified second source-fold seed; no selection using target outcomes."""
from pathlib import Path
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

O = P / 'independent_test'


def main():
    f = pd.read_parquet(O / 'features.parquet')
    original = pd.read_parquet(O / 'frozen_target_predictions.parquet')
    pred = original[['name', 'aa']].copy()
    receipts = []
    for source in ['GFP_MP', 'MP_mCherry']:
        d = b.A[b.A.source_sheet.eq(source) & b.A.group.eq('microprotein_WT') &
                b.A.mean_stability.notna() & ~b.A.inconsistent_parent_id]
        g, y = b.groups(d.protein_sequence), d.mean_stability.to_numpy()
        for model in ['A', 'T']:
            x = f.loc[d.protein_sequence, b.SETS[model]].to_numpy(float)
            z = f.loc[pred.aa, b.SETS[model]].to_numpy(float)
            pp, alpha, tuning = b.fit_predict(x, y, g, z, 20261032)
            pred[source + '__' + model] = pp
            receipts.append(dict(source=source, model=model, seed=20261032, alpha=alpha,
                                 source_inner_errors=tuning, target_outcomes_used_for_fitting=False))
    file = O / 'seed32_target_predictions.parquet'
    pred.to_parquet(file, index=False)
    # The source fits/predictions are committed before target association checks.
    (O / 'SEED32_FIT_RECEIPT.json').write_text(json.dumps(dict(prediction_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        fits=receipts, purpose='prespecified second-seed sensitivity; target already inspected for primary models'), indent=2) + '\n')
    evaluated = pd.read_parquet(O / 'evaluated_external_predictions.parquet')
    assert evaluated.name.equals(pred.name)
    rows = []
    for subset in ['primary60', 'sensitivity40']:
        keep = evaluated[subset + '_eligible']
        y = evaluated.loc[keep, 'abundance_score']
        for source in ['GFP_MP', 'MP_mCherry']:
            a = float(spearmanr(y, pred.loc[keep, source + '__A']).statistic)
            t = float(spearmanr(y, pred.loc[keep, source + '__T']).statistic)
            rows.append(dict(subset=subset, source=source, n=int(keep.sum()), A_rho=a, T_rho=t, A_minus_T=a-t))
    pd.DataFrame(rows).to_csv(O / 'seed32_sensitivity.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    with b.threadpool_limits(limits=4):
        main()
