"""Retrospective same-peptide reporter-context benchmark; see PROTOCOL.txt."""
from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

P = Path(__file__).resolve().parent
INPUT = P / 'inputs'
OUT = P / 'results'
OUT.mkdir(exist_ok=True)
SEED = 20261031
ALPHAS = np.array([.1, 1, 10, 100, 1000])
CONTEXTS = dict(GFP_original='GFP_MP', mCherry_original='MP_mCherry',
                GFP_mutagenesis='GFP_MP_mutagenesis')
receipts = []
warnings.filterwarnings('ignore', message='An input array is constant')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rho(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3 or np.std(p[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(spearmanr(y[ok], p[ok]).statistic)


def folds(g, n=5, seed=SEED):
    u, c = np.unique(g, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(u))
    order = order[np.argsort(-c[order], kind='stable')]
    size = np.zeros(n, int)
    lookup = {}
    for j in order:
        k = int(size.argmin())
        lookup[u[j]] = k
        size[k] += c[j]
    return np.array([lookup[v] for v in g])


def path_predict(x, y, z, alphas=ALPHAS):
    mean, sd = x.mean(0), x.std(0)
    sd[sd < 1e-12] = 1
    xx, zz = (x - mean) / sd, (z - mean) / sd
    ym = float(y.mean())
    v, q = eigh(xx.T @ xx, check_finite=False)
    coefs = q @ ((q.T @ (xx.T @ (y - ym)))[:, None] /
                  (np.maximum(v, 0)[:, None] + np.asarray(alphas)[None, :]))
    return zz @ coefs + ym


def fitted_prediction(x, y, g, z, seed):
    inner = folds(g, 3, seed)
    err = np.zeros(len(ALPHAS))
    for k in range(3):
        tr, te = inner != k, inner == k
        assert not set(g[tr]) & set(g[te])
        err += abs(y[te, None] - path_predict(x[tr], y[tr], x[te])).sum(0)
    alpha = float(ALPHAS[int(err.argmin())])
    return path_predict(x, y, z, [alpha])[:, 0], alpha


def cross_predictions(x, y, g, seed):
    # These are used only to calibrate within an outer training partition.
    ff = folds(g, 3, seed)
    out = np.full(len(y), np.nan)
    for k in range(3):
        tr, te = ff != k, ff == k
        assert not set(g[tr]) & set(g[te])
        out[te], _ = fitted_prediction(x[tr], y[tr], g[tr], x[te], seed + k + 31)
    assert np.isfinite(out).all()
    return out


def isotonic(s, y, z):
    increasing = rho(y, s) >= 0
    model = IsotonicRegression(increasing=increasing, out_of_bounds='clip').fit(s, y)
    return model.predict(z), bool(increasing)


def prepare():
    a = pd.read_parquet(INPUT / 'natural_assay_records.parquet')
    pieces = {}
    audit = []
    keys = ['protein_identifier', 'protein_sequence']
    for ctx, sheet in CONTEXTS.items():
        d = a[a.source_sheet.eq(sheet) & a.group.eq('microprotein_WT')].copy()
        assert not d.duplicated(keys).any()
        assert not d.protein_sequence.duplicated().any()
        valid = d.mean_stability.notna() & ~d.inconsistent_parent_id
        audit.append(dict(context=ctx, source_sheet=sheet, released_WT=len(d),
                          measured_WT=int(d.mean_stability.notna().sum()),
                          consistent_measured_WT=int(valid.sum())))
        pieces[ctx] = d.loc[valid, keys + ['mean_stability', 'stability_r1', 'stability_r2']].rename(
            columns={c: ctx + '__' + c for c in ['mean_stability', 'stability_r1', 'stability_r2']})
    pair_rows = []
    for i, s in enumerate(CONTEXTS):
        for t in list(CONTEXTS)[i + 1:]:
            z = pieces[s].merge(pieces[t], on=keys, validate='one_to_one')
            pair_rows.append(dict(source=s, target=t, n=len(z),
                                  rho=rho(z[s + '__mean_stability'], z[t + '__mean_stability'])))
    pd.DataFrame(pair_rows).to_csv(OUT / 'all_available_pair_correlations.csv', index=False)
    d = None
    for p in pieces.values():
        d = p if d is None else d.merge(p, on=keys, validate='one_to_one')
    d = d.sort_values('protein_sequence').reset_index(drop=True)
    groups = pd.read_parquet(INPUT / 'sequence_dependence_groups.parquet').set_index('sequence')
    for name in ['primary_group', 'sensitivity_group']:
        d[name] = groups.loc[d.protein_sequence, name].to_numpy()
    for row in audit:
        row['complete_three_context_cohort'] = len(d)
        row['lost_from_consistent_measured'] = row['consistent_measured_WT'] - len(d)
    pd.DataFrame(audit).to_csv(OUT / 'cohort_attrition.csv', index=False)
    d.to_parquet(OUT / 'matched_measurements.parquet', index=False)
    return d


def bootstrap_indices(g, n=1000):
    u = np.unique(g)
    members = [np.flatnonzero(g == v) for v in u]
    rng = np.random.default_rng(20261101)
    return [np.concatenate([members[j] for j in rng.integers(0, len(u), len(u))]) for _ in range(n)]


def interval(values):
    v = np.asarray(values, float)
    return np.quantile(v[np.isfinite(v)], [.025, .975]).tolist()


def observed_agreement(d):
    ix = bootstrap_indices(d.primary_group.to_numpy())
    rows = []
    for i, s in enumerate(CONTEXTS):
        for t in list(CONTEXTS)[i + 1:]:
            a, b = d[s + '__mean_stability'].to_numpy(), d[t + '__mean_stability'].to_numpy()
            lo, hi = interval([rho(a[j], b[j]) for j in ix])
            rows.append(dict(kind='between_context', source=s, target=t, n=len(d),
                             rho=rho(a, b), rho_low=lo, rho_high=hi))
    for ctx in CONTEXTS:
        a, b = d[ctx + '__stability_r1'].to_numpy(), d[ctx + '__stability_r2'].to_numpy()
        lo, hi = interval([rho(a[j], b[j]) for j in ix])
        rows.append(dict(kind='within_context_replicates', source=ctx, target=ctx,
                         n=int((np.isfinite(a) & np.isfinite(b)).sum()),
                         rho=rho(a, b), rho_low=lo, rho_high=hi))
    pd.DataFrame(rows).to_csv(OUT / 'observed_agreement.csv', index=False)


def run_setting(d, feature, sets, name, group_col, seed, with_intervals):
    g = d[group_col].to_numpy()
    ff = folds(g, seed=seed)
    out = d[['protein_identifier', 'protein_sequence', group_col]].copy()
    out['eval_group'], out['fold'] = g, ff
    pred, cal, bridges, means = {}, {}, {}, {}
    for ctx in CONTEXTS:
        y = d[ctx + '__mean_stability'].to_numpy()
        out[ctx + '__observed'] = y
        means[ctx] = np.empty(len(d))
        for k in range(5):
            means[ctx][ff == k] = y[ff != k].mean()
        for model in ['A', 'T']:
            print(name, ctx, model, flush=True)
            x = feature.loc[d.protein_sequence, sets[model]].to_numpy(float)
            pp = np.full(len(d), np.nan)
            for k in range(5):
                tr, te = ff != k, ff == k
                assert not set(g[tr]) & set(g[te])
                pp[te], alpha = fitted_prediction(x[tr], y[tr], g[tr], x[te], seed + k + 19)
                # Calibration data are generated without their source group's labels.
                oof_source = cross_predictions(x[tr], y[tr], g[tr], seed + k + 101)
                receipts.append(dict(setting=name, source=ctx, model=model, fold=k,
                                     train_rows=int(tr.sum()), test_rows=int(te.sum()),
                                     alpha=alpha, overlapping_groups=0,
                                     calibration_source_predictions='inner group out-of-fold'))
                for target in CONTEXTS:
                    if target == ctx:
                        continue
                    ty = d[target + '__mean_stability'].to_numpy()
                    key = (ctx, target, model)
                    if key not in cal:
                        cal[key] = np.full(len(d), np.nan)
                    cal[key][te], direction = isotonic(oof_source, ty[tr], pp[te])
                    receipts.append(dict(setting=name, source=ctx, target=target,
                                         model=model, fold=k, calibration_increasing=direction,
                                         target_test_rows_used_for_calibration=0))
            assert np.isfinite(pp).all()
            pred[ctx, model] = pp
            out[ctx + '__' + model] = pp
    for source in CONTEXTS:
        for target in CONTEXTS:
            if source == target:
                continue
            s = d[source + '__mean_stability'].to_numpy()
            y = d[target + '__mean_stability'].to_numpy()
            q = np.full(len(d), np.nan)
            for k in range(5):
                tr, te = ff != k, ff == k
                q[te], _ = isotonic(s[tr], y[tr], s[te])
            bridges[source, target] = q
            out[source + '__to__' + target + '__measured_bridge'] = q
            for model in ['A', 'T']:
                out[source + '__to__' + target + '__' + model + '_calibrated'] = cal[source, target, model]
    out.to_parquet(OUT / (name + '_predictions.parquet'), index=False)
    ix = bootstrap_indices(g) if with_intervals else []
    metrics, contrasts = [], []
    for source in CONTEXTS:
        for target in CONTEXTS:
            if source == target:
                continue
            y = d[target + '__mean_stability'].to_numpy()
            for model in ['A', 'T']:
                values = dict(source_sequence_raw=pred[source, model],
                              source_sequence_calibrated=cal[source, target, model],
                              target_sequence=pred[target, model],
                              measured_source_bridge=bridges[source, target],
                              training_target_mean=means[target])
                for method, p in values.items():
                    raw = method == 'source_sequence_raw'
                    r = dict(setting=name, source=source, target=target, model=model,
                             method=method, n=len(d), groups=len(np.unique(g)), rho=rho(y, p),
                             mae=np.nan if raw else float(abs(y - p).mean()))
                    if ix:
                        r['rho_low'], r['rho_high'] = interval([rho(y[j], p[j]) for j in ix])
                        if not raw:
                            r['mae_low'], r['mae_high'] = interval([float(abs(y[j] - p[j]).mean()) for j in ix])
                    metrics.append(r)
                source_p, target_p = pred[source, model], pred[target, model]
                calibrated_p = cal[source, target, model]
                r = dict(setting=name, source=source, target=target, model=model,
                         target_minus_source_rho=rho(y, target_p) - rho(y, source_p),
                         calibrated_transfer_minus_target_mae=float((abs(y - calibrated_p) - abs(y - target_p)).mean()))
                if ix:
                    r['rho_gap_low'], r['rho_gap_high'] = interval([rho(y[j], target_p[j]) - rho(y[j], source_p[j]) for j in ix])
                    r['mae_gap_low'], r['mae_gap_high'] = interval([float((abs(y[j] - calibrated_p[j]) - abs(y[j] - target_p[j])).mean()) for j in ix])
                contrasts.append(r)
    return metrics, contrasts


def main():
    freeze = json.loads((P / 'INPUT_FREEZE.json').read_text())
    for item in freeze['inputs']:
        assert sha(INPUT / item['file']) == item['sha256']
    d = prepare()
    observed_agreement(d)
    f = pd.read_parquet(INPUT / 'sequence_features.parquet').set_index('sequence')
    sets = json.loads((INPUT / 'FEATURE_DEFINITION.json').read_text())['feature_sets']
    # Check the reusable fast solver independently against sklearn.
    x = f.loc[d.protein_sequence, sets['T']].to_numpy(float)
    y = d['GFP_original__mean_stability'].to_numpy()
    ff = folds(d.primary_group.to_numpy())
    tr, te = ff != 0, ff == 0
    scaled = StandardScaler().fit(x[tr])
    errors = []
    for alpha in ALPHAS:
        reference = Ridge(alpha=alpha, solver='cholesky').fit(scaled.transform(x[tr]), y[tr]).predict(scaled.transform(x[te]))
        errors.append(float(np.max(abs(reference - path_predict(x[tr], y[tr], x[te], [alpha])[:, 0]))))
    assert max(errors) < 1e-8
    write_json('SOLVER_CHECK.json', dict(alphas=ALPHAS.tolist(), max_absolute_errors=errors))
    metrics, contrasts = [], []
    for name, groups, seed, ci in [('primary', 'primary_group', SEED, True),
                                  ('identity40', 'sensitivity_group', SEED, False),
                                  ('seed32', 'primary_group', 20261032, False)]:
        m, c = run_setting(d, f, sets, name, groups, seed, ci)
        metrics.extend(m)
        contrasts.extend(c)
    pd.DataFrame(metrics).to_csv(OUT / 'transfer_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(OUT / 'paired_transfer_contrasts.csv', index=False)
    write_json('fit_and_calibration_receipts.json', receipts)
    write_json('ANALYSIS_COMPLETE.json', dict(status='complete', common_rows=len(d),
               primary_groups=int(d.primary_group.nunique()), sensitivity_groups=int(d.sensitivity_group.nunique()),
               all_six_directions_retained=True, target_test_labels_used_in_fitting=False,
               study_status='retrospective exploratory; same publication, paired constructs',
               bootstrap='1000 group draws conditional on saved predictions; not retraining intervals'))
    print('Complete', len(d), 'matched peptides', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
