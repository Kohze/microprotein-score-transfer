"""Evaluate frozen predictions once the outcome-free separation is terminal."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path(__file__).resolve().parent
O = P / 'independent_test'


def rho(y, p):
    return float(spearmanr(y, p).statistic)


def main():
    separation = json.loads((P / 'INDEPENDENT_SEPARATION_AUDIT.json').read_text())
    freeze = json.loads((O / 'PREDICTION_FREEZE.json').read_text())
    predictions = O / 'frozen_target_predictions.parquet'
    assert hashlib.sha256(predictions.read_bytes()).hexdigest() == freeze['prediction_sha256']
    metadata = pd.read_parquet(P / 'independent_data/separated_author_holdout.parquet')
    pred = pd.read_parquet(predictions)
    assert pred.name.equals(metadata.name)
    source = pd.read_csv(P / 'independent_data/Voutsinos_degron_scores.csv.gz', usecols=['name', 'aa', 'abundance_score'])
    d = pred.merge(metadata[['name', 'source_gene_overlap', 'primary60_eligible', 'sensitivity40_eligible']], on='name', validate='one_to_one')
    d = d.merge(source, on=['name', 'aa'], validate='one_to_one')
    assert len(d) == freeze['target_rows'] and d.abundance_score.notna().all()
    models = [c for c in pred if c.startswith(('GFP_MP__', 'MP_mCherry__', 'PAP_'))]
    rows, contrasts = [], []
    for subset in ['primary60', 'sensitivity40']:
        z = d[d[subset + '_eligible']].reset_index(drop=True)
        y = z.abundance_score.to_numpy()
        g = z.protein_id.to_numpy()
        for model in models:
            rows.append(dict(subset=subset, model=model, n=len(z), genes=int(len(np.unique(g))),
                             rho=rho(y, z[model].to_numpy())))
        members = [np.flatnonzero(g == value) for value in np.unique(g)]
        rng = np.random.default_rng(20261101)
        boot = [np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))]) for _ in range(1000)]
        for source_context in ['GFP_MP', 'MP_mCherry']:
            aa, tt = z[source_context + '__A'].to_numpy(), z[source_context + '__T'].to_numpy()
            a_boot = np.array([rho(y[i], aa[i]) for i in boot])
            t_boot = np.array([rho(y[i], tt[i]) for i in boot])
            interval = np.quantile(a_boot - t_boot, [.025, .975])
            contrasts.append(dict(subset=subset, source=source_context, contrast='A_minus_T_rho',
                n=len(z), genes=len(members), estimate=rho(y, aa)-rho(y, tt), low=interval[0], high=interval[1],
                design='exploratory independent-study transfer; conditional gene bootstrap'))
            for model, values in [('A', a_boot), ('T', t_boot)]:
                row = next(r for r in rows if r['subset'] == subset and r['model'] == source_context + '__' + model)
                row['low'], row['high'] = np.quantile(values, [.025, .975])
            print(subset, source_context, contrasts[-1], flush=True)
    d.to_parquet(O / 'evaluated_external_predictions.parquet', index=False)
    pd.DataFrame(rows).to_csv(O / 'external_metrics.csv', index=False)
    pd.DataFrame(contrasts).to_csv(O / 'external_paired_contrasts.csv', index=False)
    report = dict(status='external transfer evaluated; interpretation and verification still required',
        independent_source='Voutsinos canonical cytosolic peptide tiles; not full-length nORF validation',
        new_target_used_for_model_fitting=False, target_dependent_reversal_or_model_selection=False,
        prediction_freeze_sha256=hashlib.sha256((O / 'PREDICTION_FREEZE.json').read_bytes()).hexdigest(),
        published_vs_released_count='Paper 15,815; official data/model_data.out 15,818; all released valid test members retained before exclusions',
        subgroup_and_feature_diagnostics='all retained, secondary; only A/T primary contrasts have conditional bootstrap intervals',
        uncertainty='1000 gene-cluster resamples of frozen predictions, not full retraining intervals')
    (O / 'EVALUATION_RECEIPT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
