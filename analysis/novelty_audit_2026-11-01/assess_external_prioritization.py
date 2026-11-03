"""Descriptive top-decile policy consequences of the frozen predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P / 'independent_test'


def main():
    d = pd.read_parquet(O / 'evaluated_external_predictions.parquet')
    d = d[d.primary60_eligible].sort_values('aa').reset_index(drop=True)
    models = [c for c in d if c.startswith(('GFP_MP__', 'MP_mCherry__', 'PAP_'))]
    n = int(np.ceil(.1*len(d)))
    y = d.abundance_score.to_numpy()
    true = np.zeros(len(d), bool)
    true[np.argsort(-y, kind='stable')[:n]] = True
    rows = []
    for model in models:
        selected = np.argsort(-d[model].to_numpy(), kind='stable')[:n]
        rows.append(dict(model=model, n=len(d), selected=n,
            assay_top_decile_precision=float(true[selected].mean()),
            selected_mean_assay_score=float(y[selected].mean()),
            selected_gene_count=int(d.iloc[selected].protein_id.nunique()),
            interpretation='assay rank prioritization; not stable/function classification'))
    pd.DataFrame(rows).to_csv(O / 'top_decile_prioritization.csv', index=False)
    g = d.protein_id.to_numpy()
    members = [np.flatnonzero(g==v) for v in np.unique(g)]
    rng = np.random.default_rng(20261102)
    diffs = {s: [] for s in ['GFP_MP', 'MP_mCherry']}
    for _ in range(1000):
        ix = np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
        yy = y[ix]
        take = int(np.ceil(.1*len(ix)))
        for s in diffs:
            ia = np.argsort(-d[s+'__A'].to_numpy()[ix], kind='stable')[:take]
            it = np.argsort(-d[s+'__T'].to_numpy()[ix], kind='stable')[:take]
            diffs[s].append(float(yy[ia].mean()-yy[it].mean()))
    contrast = []
    for s, values in diffs.items():
        aa = next(r for r in rows if r['model']==s+'__A')
        tt = next(r for r in rows if r['model']==s+'__T')
        low, high = np.quantile(values,[.025,.975])
        contrast.append(dict(source=s, contrast='A minus T: mean assay score among each policy top decile',
            estimate=aa['selected_mean_assay_score']-tt['selected_mean_assay_score'],low=low,high=high))
    pd.DataFrame(contrast).to_csv(O/'top_decile_policy_contrasts.csv', index=False)
    (O/'PRIORITIZATION_SCOPE.json').write_text(json.dumps(dict(
        status='completed descriptive policy extension', policy='top ceil(10%) by each frozen score; ties lexical peptide order',
        primary_fraction_basis='same 10% fraction as prior benchmark prioritization; not tuned on this target',
        target_definition='top ceil(10%) measured assay ranks with same tie rule; not biological stable/function labels',
        uncertainty='1000 conditional gene-bootstrap draws; selection recalculated per draw',
        limitation='canonical tile assay, retrospective selection-rule analysis, not prospective candidate validation'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(contrast).to_string(index=False))


if __name__ == '__main__':
    main()
