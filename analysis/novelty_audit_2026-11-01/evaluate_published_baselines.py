"""Challenge the novelty case with frozen published composition predictors."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path(__file__).resolve().parent
O = P/'published_predictor_comparison'
B = P.parent/'stability_benchmark_2026-10-31'
MODELS = ['QCDPred_mean17_stability','QCDPred_normalized_cargo_stability',
          'human25_first25_stability','human25_normalized_cargo_stability']


def rho(y,p):
    valid = np.isfinite(y)&np.isfinite(p)
    return float(spearmanr(y[valid],p[valid]).statistic)


def selected_mean(y,p):
    n = int(np.ceil(.1*len(y)))
    return float(y[np.argsort(-p,kind='stable')[:n]].mean())


def main():
    freeze = json.loads((O/'BASELINE_PREDICTION_FREEZE.json').read_text())
    file = O/'frozen_published_predictions.parquet'
    assert hashlib.sha256(file.read_bytes()).hexdigest() == freeze['prediction_sha256']
    pred = pd.read_parquet(file)
    target = pd.read_parquet(P/'independent_test/evaluated_external_predictions.parquet')
    d = target.merge(pred,left_on='aa',right_on='sequence',validate='one_to_one')
    assert len(d) == len(target)
    d.to_parquet(O/'evaluated_published_predictions.parquet',index=False)
    rows,policy,contrasts = [],[],[]
    comparison = ['GFP_MP__A','GFP_MP__T','MP_mCherry__A','MP_mCherry__T']
    for subset in ['primary60','sensitivity40']:
        z = d[d[subset+'_eligible']].sort_values('aa').reset_index(drop=True)
        y = z.abundance_score.to_numpy()
        for model in MODELS+comparison:
            p = z[model].to_numpy()
            rows.append(dict(cohort='Voutsinos_'+subset,model=model,n=int(np.isfinite(p).sum()),
                genes=int(z.protein_id.nunique()),rho=rho(y,p),
                exact_training_window_rows=int(z.exact_labeled_QCDPred_training_window.sum())))
            policy.append(dict(subset=subset,model=model,n=len(z),selected=int(np.ceil(.1*len(z))),
                selected_mean_assay_score=selected_mean(y,p)))
        # All these are explicitly secondary conditional paired contrasts.
        if subset == 'primary60':
            groups = z.protein_id.to_numpy()
            members = [np.flatnonzero(groups==g) for g in np.unique(groups)]
            rng = np.random.default_rng(20261103)
            delta = {m:[] for m in comparison}
            pdiff = {m:[] for m in comparison}
            qcd = z.QCDPred_mean17_stability.to_numpy()
            comp = {m:z[m].to_numpy() for m in comparison}
            for draw in range(1000):
                ix = np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
                yy,qq = y[ix],qcd[ix]
                qr,qs = rho(yy,qq),selected_mean(yy,qq)
                for model,values in comp.items():
                    delta[model].append(qr-rho(yy,values[ix]))
                    pdiff[model].append(qs-selected_mean(yy,values[ix]))
                if (draw+1)%250 == 0:
                    print('conditional bootstrap',draw+1,flush=True)
            for model in comparison:
                for metric,values,estimate in [
                    ('rho',delta[model],rho(y,qcd)-rho(y,comp[model])),
                    ('selected_mean',pdiff[model],selected_mean(y,qcd)-selected_mean(y,comp[model]))]:
                    lo,hi = np.quantile(values,[.025,.975])
                    contrasts.append(dict(subset=subset,contrast='QCDPred_mean17 minus '+model,
                        metric=metric,estimate=estimate,low=lo,high=hi,
                        interpretation='secondary conditional gene bootstrap; no multiplicity-adjusted claim'))
    guo = pd.read_parquet(B/'natural_assay_records.parquet')
    guo = guo[guo.group.eq('microprotein_WT') & guo.source_sheet.isin(['GFP_MP','MP_mCherry']) &
              guo.mean_stability.notna() & ~guo.inconsistent_parent_id]
    guo = guo.merge(pred,left_on='protein_sequence',right_on='sequence',validate='many_to_one')
    for source,z in guo.groupby('source_sheet'):
        y = z.mean_stability.to_numpy()
        for model in MODELS:
            values = z[model].to_numpy()
            rows.append(dict(cohort='Guo_'+source,model=model,n=int(np.isfinite(values).sum()),
                rho=rho(y,values),exact_training_window_rows=int(z.exact_labeled_QCDPred_training_window.sum())))
    pd.DataFrame(rows).to_csv(O/'published_baseline_metrics.csv',index=False)
    pd.DataFrame(policy).to_csv(O/'published_baseline_policies.csv',index=False)
    pd.DataFrame(contrasts).to_csv(O/'published_baseline_contrasts.csv',index=False)
    result = dict(status='completed; interpretation and manuscript implications pending',
        freeze_sha256=hashlib.sha256((O/'BASELINE_PREDICTION_FREEZE.json').read_bytes()).hexdigest(),
        target_outcome_training=False,new_experimental_data=False,
        primary_mean17_aggregation_fixed_before_this_comparison=True,
        target_A_T_outcomes_already_seen=True,
        independent_scope='QCDPred is trained on yeast; human25 is target-study trained',
        training_overlap='Exact labeled yeast 17mer windows audited; this is not a homology-separation guarantee',
        conditional_bootstrap_draws=1000)
    (O/'BASELINE_EVALUATION_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
