"""Retain normalized-score contrasts and the pre-existing strict additional cohort."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_published_baselines import rho,selected_mean,MODELS

P=Path(__file__).resolve().parent
O=P/'published_predictor_comparison'
B=P.parent/'stability_benchmark_2026-10-31'


def main():
    frozen=pd.read_parquet(O/'frozen_published_predictions.parquet')
    additional=pd.read_parquet(B/'results/additional_transfer.parquet')
    z=additional[additional.strict_disjoint].merge(frozen,left_on='protein_sequence',right_on='sequence',validate='one_to_one')
    assert len(z)==602
    rows=[]
    for model in MODELS:
        y,p=z.mean_stability.to_numpy(),z[model].to_numpy()
        rows.append(dict(cohort='Guo_additional_strict602',model=model,n=int(np.isfinite(p).sum()),rho=rho(y,p),
            exact_training_window_rows=int(z.exact_labeled_QCDPred_training_window.sum())))
    pd.DataFrame(rows).to_csv(O/'additional_strict602_baselines.csv',index=False)
    d=pd.read_parquet(O/'evaluated_published_predictions.parquet')
    d=d[d.primary60_eligible].sort_values('aa').reset_index(drop=True)
    q=d.QCDPred_normalized_cargo_stability.to_numpy()
    y=d.abundance_score.to_numpy()
    comparison={m:d[m].to_numpy() for m in ['GFP_MP__A','GFP_MP__T','MP_mCherry__A','MP_mCherry__T']}
    groups=d.protein_id.to_numpy()
    members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    rng=np.random.default_rng(20261103)
    draws={m:dict(rho=[],selected_mean=[]) for m in comparison}
    for draw in range(1000):
        ix=np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
        yy,qq=y[ix],q[ix]
        r,s=rho(yy,qq),selected_mean(yy,qq)
        for model,values in comparison.items():
            draws[model]['rho'].append(r-rho(yy,values[ix]))
            draws[model]['selected_mean'].append(s-selected_mean(yy,values[ix]))
        if (draw+1)%250==0:
            print('normalized representation bootstrap',draw+1,flush=True)
    rows=[]
    for model,stats in draws.items():
        for metric,values in stats.items():
            lo,hi=np.quantile(values,[.025,.975])
            est=(rho(y,q)-rho(y,comparison[model])) if metric=='rho' else (selected_mean(y,q)-selected_mean(y,comparison[model]))
            rows.append(dict(contrast='QCDPred normalized cargo minus '+model,metric=metric,estimate=est,low=lo,high=hi,
                design='secondary predefined representation; conditional gene bootstrap; no multiplicity-adjusted claim'))
    pd.DataFrame(rows).to_csv(O/'normalized_representation_contrasts.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()
