"""Check new outputs against the original workbook and an independent fit."""
from pathlib import Path
import analyze_paired_context as a
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

P = Path(__file__).resolve().parent
R = P / 'results'


def main():
    d = pd.read_parquet(R / 'matched_measurements.parquet')
    workbook = P / 'inputs/Guo2026_mmc4.xlsx'
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == '716ebbb435ec058d43066b2ba1ac034d377814a341e5e39b342aaa36ce23691c'
    checked = 0
    for ctx, sheet in a.CONTEXTS.items():
        original = pd.read_excel(workbook, sheet_name=sheet)
        joined = d.merge(original, on=['protein_identifier', 'protein_sequence'], validate='one_to_one')
        assert len(joined) == len(d)
        for c in ['mean_stability', 'stability_r1', 'stability_r2']:
            assert np.allclose(joined[ctx + '__' + c], joined[c], equal_nan=True, atol=1e-14, rtol=0)
            checked += len(d)
    metric = pd.read_csv(R / 'transfer_metrics.csv')
    max_rho_error = max_mae_error = 0.
    for setting in ['primary', 'identity40', 'seed32']:
        p = pd.read_parquet(R / (setting + '_predictions.parquet'))
        assert p.protein_sequence.equals(d.protein_sequence)
        assert p.groupby('eval_group').fold.nunique().eq(1).all()
        assert set(p.fold) == set(range(5))
        for ctx in a.CONTEXTS:
            assert np.array_equal(p[ctx + '__observed'], d[ctx + '__mean_stability'])
        for row in metric[metric.setting.eq(setting)].itertuples():
            y = p[row.target + '__observed'].to_numpy()
            if row.method == 'source_sequence_raw':
                v = p[row.source + '__' + row.model].to_numpy()
            elif row.method == 'source_sequence_calibrated':
                v = p[row.source + '__to__' + row.target + '__' + row.model + '_calibrated'].to_numpy()
            elif row.method == 'target_sequence':
                v = p[row.target + '__' + row.model].to_numpy()
            elif row.method == 'measured_source_bridge':
                v = p[row.source + '__to__' + row.target + '__measured_bridge'].to_numpy()
            else:
                v = np.array([y[p.fold.ne(k)].mean() for k in p.fold])
            rr = float(spearmanr(y, v).statistic)
            max_rho_error = max(max_rho_error, abs(rr - row.rho))
            if row.method != 'source_sequence_raw':
                max_mae_error = max(max_mae_error, abs(float(abs(y - v).mean()) - row.mae))
        for source in a.CONTEXTS:
            for target in a.CONTEXTS:
                if source == target:
                    continue
                expected = np.empty(len(p))
                s, y = p[source + '__observed'].to_numpy(), p[target + '__observed'].to_numpy()
                for k in range(5):
                    tr, te = p.fold.ne(k).to_numpy(), p.fold.eq(k).to_numpy()
                    model = IsotonicRegression(increasing=spearmanr(s[tr], y[tr]).statistic >= 0,
                                               out_of_bounds='clip').fit(s[tr], y[tr])
                    expected[te] = model.predict(s[te])
                assert np.max(abs(expected - p[source + '__to__' + target + '__measured_bridge'])) < 1e-12
    assert max_rho_error < 1e-12 and max_mae_error < 1e-12
    # Fresh sklearn implementation, including independent alpha selection, for
    # one complete outer fold. It does not reuse the eigensolver for this fit.
    p = pd.read_parquet(R / 'primary_predictions.parquet')
    tr, te = p.fold.ne(0).to_numpy(), p.fold.eq(0).to_numpy()
    f = pd.read_parquet(P / 'inputs/sequence_features.parquet').set_index('sequence')
    cols = json.loads((P / 'inputs/FEATURE_DEFINITION.json').read_text())['feature_sets']['T']
    x, y, g = f.loc[d.protein_sequence, cols].to_numpy(float), p.GFP_original__observed.to_numpy(), p.eval_group.to_numpy()
    inner = a.folds(g[tr], 3, a.SEED + 19)
    errors = []
    for alpha in a.ALPHAS:
        e = 0.
        for k in range(3):
            fit, val = inner != k, inner == k
            scale = StandardScaler().fit(x[tr][fit])
            model = Ridge(alpha=alpha, solver='cholesky').fit(scale.transform(x[tr][fit]), y[tr][fit])
            e += abs(model.predict(scale.transform(x[tr][val])) - y[tr][val]).sum()
        errors.append(float(e))
    alpha = float(a.ALPHAS[int(np.argmin(errors))])
    scale = StandardScaler().fit(x[tr])
    model = Ridge(alpha=alpha, solver='cholesky').fit(scale.transform(x[tr]), y[tr])
    fit_error = float(np.max(abs(model.predict(scale.transform(x[te])) - p.loc[te, 'GFP_original__T'])))
    assert fit_error < 1e-8
    rec = json.loads((R / 'fit_and_calibration_receipts.json').read_text())
    assert all(r.get('overlapping_groups', 0) == 0 for r in rec)
    assert all(r.get('target_test_rows_used_for_calibration', 0) == 0 for r in rec)
    result = dict(status='passed', original_workbook_values_checked=checked,
                  common_rows=len(d), strict_join_keys=['identifier', 'exact peptide'],
                  settings=3, all_saved_metrics_recomputed=True,
                  max_metric_rho_error=max_rho_error, max_metric_mae_error=max_mae_error,
                  measured_source_bridges_independently_refit=18,
                  sklearn_outer_fold_selected_alpha=alpha, sklearn_refit_max_error=fit_error,
                  primary_five_fold_group_sizes=p.groupby('fold').size().to_dict())
    (R / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    with a.threadpool_limits(limits=4):
        main()
