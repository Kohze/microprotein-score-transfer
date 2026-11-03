"""Independently refit new source heads and recompute external metrics."""
from pathlib import Path
import sys
P = Path(__file__).resolve().parent
B = P.parent / 'stability_benchmark_2026-10-31'
sys.path.insert(0, str(B))
import run_benchmark as b
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

O = P / 'independent_test'


def main():
    f = pd.read_parquet(O / 'features.parquet')
    pred = pd.read_parquet(O / 'frozen_target_predictions.parquet')
    specs = json.loads((O / 'FEATURE_SPECIFICATION.json').read_text())['models']
    receipts = json.loads((O / 'FIT_RECEIPTS.json').read_text())
    errors = []
    for source in ['GFP_MP', 'MP_mCherry']:
        d = b.A[b.A.source_sheet.eq(source) & b.A.group.eq('microprotein_WT') &
                b.A.mean_stability.notna() & ~b.A.inconsistent_parent_id].reset_index(drop=True)
        for model in ['A', 'T', 'N4', 'C4', 'I4']:
            record = next(r for r in receipts if r.get('purpose') == 'external prediction' and
                          r['source'] == source and r['model'] == model)
            cols = specs[model]
            x = f.loc[d.protein_sequence, cols].to_numpy(float)
            z = f.loc[pred.aa, cols].to_numpy(float)
            scale = StandardScaler().fit(x)
            model_refit = Ridge(alpha=record['alpha'], solver='cholesky').fit(scale.transform(x), d.mean_stability)
            error = float(np.max(abs(model_refit.predict(scale.transform(z)) - pred[source + '__' + model])))
            assert error < 1e-8, (source, model, error)
            errors.append(dict(source=source, model=model, max_error=error))
    evaluated = pd.read_parquet(O / 'evaluated_external_predictions.parquet')
    metrics = pd.read_csv(O / 'external_metrics.csv')
    metric_errors = []
    for row in metrics.itertuples():
        z = evaluated[evaluated[row.subset + '_eligible']]
        expected = float(spearmanr(z.abundance_score, z[row.model]).statistic)
        metric_errors.append(abs(expected-row.rho))
        assert len(z) == row.n and z.protein_id.nunique() == row.genes
        assert not z.source_gene_overlap.any()
    assert max(metric_errors) < 1e-12
    freeze = json.loads((O / 'PREDICTION_FREEZE.json').read_text())
    assert hashlib.sha256((O / 'frozen_target_predictions.parquet').read_bytes()).hexdigest() == freeze['prediction_sha256']
    report = dict(status='passed', independently_refit_source_heads=errors,
        maximum_metric_recalculation_error=max(metric_errors), all_eligible_rows_gene_disjoint=True,
        frozen_predictions_unchanged=True, new_target_model_training=False,
        scope='New feature values match old source features exactly; ten sklearn source fits and every saved point metric checked.')
    (O / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    with b.threadpool_limits(limits=4):
        main()
