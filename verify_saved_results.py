"""Independent, read-only point-metric checks. No model imports or network use."""
from pathlib import Path
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
CHECKS = {}
MAX_ERROR = 0.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    global MAX_ERROR
    require(np.isclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True),
            f'{label}: calculated {actual}, recorded {expected}')
    if np.isfinite(actual) and np.isfinite(expected):
        MAX_ERROR = max(MAX_ERROR, abs(float(actual) - float(expected)))


def rho(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 3 or np.std(y[ok]) < 1e-14 or np.std(p[ok]) < 1e-14:
        return np.nan
    return float(spearmanr(y[ok], p[ok]).statistic)


def table(base, name):
    path = base / name
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)


def core():
    base = ROOT / 'analysis/stability_benchmark_2026-10-31/results'
    names = ['native_GFP_MP','native_MP_mCherry','native_GFP_MP_mutagenesis',
             'sensitivity_identity40','sensitivity_seed32','sensitivity_seed33']
    saved = {n:table(base,n+'.parquet') for n in names+['additional_transfer','Yang_validation']}
    mutations = table(base,'mutation_predictions.parquet')
    metrics = table(base,'metrics.csv')
    maes = 0
    for r in metrics.itertuples(index=False):
        if r.cohort.startswith('mutation_'):
            d = mutations[mutations.mutation_group.eq(r.cohort.removeprefix('mutation_'))]
            y = d['delta']
            if r.subset.startswith('replicate'):
                rep = r.subset.removeprefix('replicate')
                y = d['stability_r'+rep+'_mut']-d['stability_r'+rep+'_WT']
        else:
            d = saved[r.cohort]
            if r.subset == 'strict_disjoint':
                d = d[d.strict_disjoint]
            elif r.subset == 'exact_disjoint':
                d = d[~d.exact_Guo_natural_sequence_overlap]
            else:
                require(r.subset == 'all','Unknown core subset')
            y = d['untreated_mean' if r.cohort=='Yang_validation' else 'mean_stability']
        p = d[r.model]
        ok = np.isfinite(y) & np.isfinite(p)
        require(int(ok.sum()) == r.n,'Core n mismatch')
        close(rho(y,p),r.rho,'Core rho '+str((r.cohort,r.subset,r.model)))
        if pd.notna(r.mae):
            close(np.mean(abs(y[ok]-p[ok])),r.mae,'Core MAE')
            maes += 1
    for name in names:
        require(saved[name].groupby('eval_group').fold.nunique().eq(1).all(),'Group crosses folds')
    leu = mutations[mutations.mutation_group.eq('L_to_A')]
    app = mutations[mutations.mutation_group.eq('microprotein_10mer_ext')]
    require((leu.protein_sequence_WT.str.replace('L','A',regex=False)==leu.protein_sequence_mut).all(),'L-to-A identity')
    require((app.protein_sequence_WT+'DNYNEPKANQ'==app.protein_sequence_mut).all(),'Append identity')
    require(np.allclose(mutations.delta,mutations.mean_stability_mut-mutations.mean_stability_WT,rtol=0,atol=1e-12),'Within-library delta identity')
    require(len(metrics)==261 and maes==140,'Unexpected core scope')
    CHECKS['core'] = {'correlation_rows':len(metrics),'reported_MAE_rows':maes,
        'group_fold_assignments':len(names),'exact_edit_rows':len(mutations)}


def paired_context():
    base = ROOT / 'analysis/paired_context_2026-11-01/results'
    frames={k:table(base,v+'_predictions.csv') for k,v in [('primary','primary'),('identity40','identity40'),('seed32','seed32')]}
    count=maes=skipped=0
    for r in table(base,'transfer_metrics.csv').itertuples(index=False):
        d=frames[r.setting]
        y=d[r.target+'__observed']
        if r.method=='training_target_mean':
            skipped+=1
            continue
        if r.method=='source_sequence_raw': col=r.source+'__'+r.model
        elif r.method=='source_sequence_calibrated': col=r.source+'__to__'+r.target+'__'+r.model+'_calibrated'
        elif r.method=='target_sequence': col=r.target+'__'+r.model
        elif r.method=='measured_source_bridge': col=r.source+'__to__'+r.target+'__measured_bridge'
        else: raise ValueError('Unknown paired-context method: '+r.method)
        require(len(d)==r.n and d.eval_group.nunique()==r.groups,'Paired-context population')
        close(rho(y,d[col]),r.rho,'Paired-context rho')
        if pd.notna(r.mae):
            close(np.mean(abs(y-d[col])),r.mae,'Paired-context MAE')
            maes+=1
        count+=1
    CHECKS['same_peptide_context']={'correlation_rows':count,'MAE_rows':maes,'unsaved_constant_baseline_rows_excluded':skipped}


def canonical():
    base=ROOT/'analysis/novelty_audit_2026-11-01/independent_test'
    d=table(base,'evaluated_external_predictions.parquet')
    metrics=table(base,'external_metrics.csv')
    for r in metrics.itertuples(index=False):
        z=d[d[r.subset+'_eligible']]
        require(len(z)==r.n and z.protein_id.nunique()==r.genes,'Canonical population')
        close(rho(z.abundance_score,z[r.model]),r.rho,'Canonical rho')
    CHECKS['canonical_tiles']={'correlation_rows':len(metrics)}


def pair_subset(d,label):
    label=label.removeprefix('Chu_')
    if label=='primary60_common_current_transcript_match':
        return d[d.primary60_common_eligible & d.WT_matches_current_transcript_C58]
    if label.startswith('RT_length_'):
        p=d[d.primary60_common_eligible]
        masks={'RT_length_le20':p.reported_RT_length.le(20),
               'RT_length_21to57':p.reported_RT_length.between(21,57),
               'RT_length_ge58':p.reported_RT_length.ge(58)}
        return p[masks[label]]
    return d[d[label+'_eligible']]


def independent_pairs():
    base=ROOT/'analysis/novelty_audit_2026-11-01'
    d=table(base/'independent_interventions','evaluated_predictions.parquet')
    metrics=table(base/'independent_interventions','transfer_metrics.csv')
    for r in metrics.itertuples(index=False):
        z=pair_subset(d,r.cohort)
        require(len(z)==r.n and z.target_dependence_group.nunique()==r.groups,'Paired population')
        for outcome,pcol,field in [('wt_mean','__wt','native_wt_rho'),('rt_mean','__rt','native_rt_rho'),
                ('observed_delta','__delta','delta_rho'),('observed_delta_r1','__delta','delta_r1_rho'),
                ('observed_delta_r2','__delta','delta_r2_rho')]:
            close(rho(z[outcome],z[r.model+pcol]),getattr(r,field),'Independent pairs '+field)
    neural=table(base/'pap_neural_comparison','paired_delta_predictions.parquet')
    require(neural.source_excel_row.is_unique and d.source_excel_row.is_unique,'Repeated source row')
    merged=d.merge(neural,on='source_excel_row',validate='one_to_one')
    nm=table(base/'pap_neural_comparison','paired_effect_metrics.csv')
    checked=0
    for r in nm.itertuples(index=False):
        z=pair_subset(merged,r.cohort)
        require(len(z)==r.n and z.target_dependence_group.nunique()==r.groups,'Neural population')
        for outcome,field in [('observed_delta','delta_rho'),('observed_delta_r1','delta_r1_rho'),('observed_delta_r2','delta_r2_rho')]:
            if pd.notna(getattr(r,field)):
                close(rho(z[outcome],z[r.model]),getattr(r,field),'Neural '+field)
                checked+=1
    CHECKS['independent_paired_effects']={'metric_rows':len(metrics),'correlations':len(metrics)*5,
        'neural_metric_rows':len(nm),'neural_delta_correlations':checked,'one_to_one_source_rows':len(merged)}


def terminal():
    base=ROOT/'analysis/novelty_audit_2026-11-01/terminal_predictor_comparison'
    d=table(base,'evaluated_window_predictions.parquet')
    for name in ['transfer_metrics.csv','window_control_metrics.csv']:
        m=table(base,name)
        for r in m.itertuples(index=False):
            z=d[d[r.cohort+'_eligible']]
            require(len(z)==r.n and z.target_dependence_group.nunique()==r.groups,'Exposure population')
            for outcome,pcol,field in [('wt_mean','__wt','native_WT_rho'),('rt_mean','__rt','native_RT_rho'),
                    ('observed_delta','__delta','delta_rho'),('observed_delta_r1','__delta','replicate1_delta_rho'),
                    ('observed_delta_r2','__delta','replicate2_delta_rho')]:
                close(rho(z[outcome],z[r.model+pcol]),getattr(r,field),'Terminal '+field)
        CHECKS[name.removesuffix('.csv')]={'metric_rows':len(m),'correlations':len(m)*5}
    counts={k:int(d[k+'_eligible'].sum()) for k in ['all1743_provenance_limited','either_endpoint_screen_matched',
        'neither_endpoint_screen_matched','neither_endpoint_or_available_symbol_overlap']}
    require(list(counts.values())==[1743,1454,289,97],'Exposure-stratum counts')
    CHECKS['terminal_exposure_population_counts']=counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for n,rec in manifest['files'].items():
        path=(ROOT/n).resolve()
        require(path.is_relative_to(ROOT.resolve()),'Unsafe manifest path')
        b=path.read_bytes()
        require(len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'],'Hash/size mismatch: '+n)
    CHECKS['manifest_members']=len(manifest['files'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=RuntimeWarning)
        core(); paired_context(); canonical(); independent_pairs(); terminal()
    result={'status':'passed','absolute_tolerance':1e-12,'maximum_absolute_metric_error':MAX_ERROR,
        'checks':CHECKS,'model_refit':False,'network_access':False,'raw_input_reconstruction':False,
        'bootstrap_interval_recomputation':False,'pretrained_inference':False,
        'scope':'Frozen point metrics and manifest only; no new validation of native biology, model training or reported confidence intervals.'}
    out=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(out,encoding='utf-8')
    print(out,end='')


if __name__=='__main__':
    main()
