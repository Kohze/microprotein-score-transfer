"""Independent rank/coverage checks for the post-hoc source-filter restriction."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import openpyxl

P = Path(__file__).resolve().parent
WS = P.parent
O = P / 'native_function_bridge'
B = WS / 'stability_benchmark_2026-10-31'


def independent_rho(a, b):
    ra, rb = pd.Series(a).rank(method='average'), pd.Series(b).rank(method='average')
    if ra.max() == ra.min() or rb.max() == rb.min():
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    receipt = json.loads((O / 'GUO_TERMINAL_FILTER_RECEIPT.json').read_text())
    predpath = B / 'results/mutation_predictions.parquet'
    assert hashlib.sha256(predpath.read_bytes()).hexdigest() == receipt['source_prediction_sha256']
    d = pd.read_parquet(predpath)
    membership = pd.read_parquet(O / 'source_terminal_filter_membership.parquet')
    wb = openpyxl.load_workbook(B / 'inputs/sources/Guo2026_mmc4.xlsx', read_only=True, data_only=True)
    iterator = wb['GFP_MP_mutagenesis'].iter_rows(values_only=True)
    header = next(iterator)
    raw = [dict(zip(header, row)) for row in iterator]
    lookup = {(r['protein_identifier'],r['group'],r['protein_sequence']):r for r in raw}
    for i, row in d.iterrows():
        for side in ['WT','mut']:
            key = (row.protein_identifier,row['group_'+side],row['protein_sequence_'+side])
            assert key in lookup
        passed = all(x != 'L' for x in row.protein_sequence_WT[-2:])
        assert passed == membership.iloc[i].source_terminal_Leu_filter_pass
    allmetrics = pd.read_csv(O / 'source_terminal_filter_metrics.csv')
    maxerr = 0
    counts = []
    for _, row in allmetrics.iterrows():
        q = d[d.mutation_group == row.family]
        mask = q.protein_sequence_WT.map(lambda s:'L' not in s[-2:])
        if row.stratum == 'source_rule_retained':
            q = q[mask]
        elif row.stratum == 'source_rule_omitted':
            q = q[~mask]
        assert len(q) == row.n and q.eval_group.nunique() == row.groups
        if row.outcome == 'mean':
            y = q.delta.to_numpy()
        else:
            rep = int(row.outcome[-1])
            y = (q[f'stability_r{rep}_mut']-q[f'stability_r{rep}_WT']).to_numpy()
        r = independent_rho(y,q[row.model])
        if np.isnan(r):
            assert np.isnan(row.rho)
        else:
            error = abs(r-row.rho)
            assert error < 1e-12
            maxerr = max(maxerr,error)
        if not pd.isna(row.MAE):
            assert abs(np.mean(abs(y-q[row.model]))-row.MAE) < 1e-12
    contrasts = pd.read_csv(O / 'source_terminal_filter_contrasts.csv')
    m = allmetrics[allmetrics.outcome.eq('mean')].set_index(['family','stratum','model'])
    for _, row in contrasts.iterrows():
        a,b = row.contrast.split(' minus ')
        expected = m.loc[(row.family,row.stratum,a),'rho']-m.loc[(row.family,row.stratum,b),'rho']
        assert abs(expected-row.estimate) < 1e-12
        assert np.isfinite(row.low) and np.isfinite(row.high) and row.low <= row.high
    out = dict(status='passed', original_workbook_pair_memberships_checked=len(d),
        metric_rows_independently_recomputed=len(allmetrics), contrast_point_estimates_checked=len(contrasts),
        max_rank_error=maxerr, frozen_prediction_hash_unchanged=True,
        bootstrap_distribution_independently_recomputed=False,
        limitations='Conditional interval implementation/source code provided; independent verification checks all metric and contrast point estimates, not a separate bootstrap replay')
    (O / 'GUO_FILTER_INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
    receipt['status'] = 'passed independent source-membership and point-metric verification'
    (O / 'GUO_TERMINAL_FILTER_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
