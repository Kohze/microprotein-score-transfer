"""Late exploratory diagnostics, separated from the frozen primary comparisons."""
import json
import numpy as np
import pandas as pd
import run_benchmark as b
from sklearn.isotonic import IsotonicRegression

def main():
    out=b.OUT;top=[]
    for name,file,strict in [('native','native_GFP_MP.parquet',False),('additional_strict','additional_transfer.parquet',True)]:
        d=pd.read_parquet(out/file)
        if strict:d=d[d.strict_disjoint]
        models=['L','A','T','O','AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop']
        if strict:models+=['Guo_regression']
        d=d.dropna(subset=models)
        for model in models:
            z=d.sort_values([model,'protein_sequence'],ascending=[False,True]);n=int(np.ceil(len(z)*.1));chosen=z.iloc[:n]
            top.append(dict(cohort=name,model=model,n_common=len(z),n_selected=n,selection='top ceil(10%); descending frozen score; ties sequence lexical',mean_outcome_selected=chosen.mean_stability.mean(),mean_outcome_all=z.mean_stability.mean(),mean_outcome_rest=z.iloc[n:].mean_stability.mean(),median_length_selected=chosen.sequence_length.median(),ratio_to_cohort_mean=chosen.mean_stability.mean()/z.mean_stability.mean()))
    pd.DataFrame(top).to_csv(out/'exploratory_top_decile_prioritization.csv',index=False)
    raw=pd.read_parquet(b.ROOT/'inputs/Guo2026_within_library_mutant_pairs.parquet');families={k:(k[0],k[-1]) for k in ['A_to_D','C_to_D','G_to_D','N_to_E','N_to_P','V_to_D']}
    rows=[]
    for r in raw[raw.mutation_group.isin(families)].to_dict('records'):
        wt,mut=r['protein_sequence_WT'],r['protein_sequence_mut'];a,c=families[r['mutation_group']]
        changes=[(i,x,y) for i,(x,y) in enumerate(zip(wt,mut)) if x!=y]
        r['specific_edit_verified']=len(wt)==len(mut) and len(changes)>0 and all(x==a and y==c for _,x,y in changes)
        r['n_changed_positions']=len(changes);r['changed_positions_1based']=';'.join(str(i+1) for i,_,_ in changes)
        r['verification_rule']='all changed positions match named source->target; unchanged positions preserved; not assumed global';rows.append(r)
    allpairs=pd.DataFrame(rows);allpairs.to_parquet(out/'exploratory_additional_edit_audit.parquet',index=False)
    d=allpairs[allpairs.specific_edit_verified & allpairs.mean_stability_mut.notna() & allpairs.mean_stability_WT.notna()].copy().reset_index(drop=True)
    d['eval_group']=b.groups(d.protein_sequence_WT);d['delta']=d.mean_stability_mut-d.mean_stability_WT
    wt=b.A[(b.A.source_sheet=='GFP_MP_mutagenesis')&(b.A.group=='microprotein_WT')&b.A.mean_stability.notna()].reset_index(drop=True);g=b.groups(wt.protein_sequence);ff=b.folds(g);d['fold']=d.eval_group.map(dict(zip(g,ff)));assert d.fold.notna().all();yy=wt.mean_stability.to_numpy()
    for model in ['T','O']:
        d[model]=np.nan;x=b.X(wt.protein_sequence,model)
        for k in range(5):
            tr=ff!=k;te=d.fold.eq(k).to_numpy();z=np.vstack([b.X(d.loc[te,'protein_sequence_WT'],model),b.X(d.loc[te,'protein_sequence_mut'],model)]);p,_,_=b.fit_predict(x[tr],yy[tr],g[tr],z,b.SEED+k+19);n=te.sum();d.loc[te,model]=p[n:]-p[:n]
    for model in ['PAP_Cterminal','PAP_full']:
        s=b.F.loc[list(wt.protein_sequence),model].to_numpy();a=b.F.loc[list(d.protein_sequence_WT),model].to_numpy();c=b.F.loc[list(d.protein_sequence_mut),model].to_numpy();d[model+'_raw']=c-a;d[model+'_calibrated']=np.nan
        for k in range(5):
            tr=ff!=k;te=d.fold.eq(k).to_numpy();m=IsotonicRegression(increasing=b.rho(yy[tr],s[tr])>=0,out_of_bounds='clip').fit(s[tr],yy[tr]);d.loc[te,model+'_calibrated']=m.predict(c[te])-m.predict(a[te])
    result=[]
    for fam,z in d.groupby('mutation_group'):
        for model in ['T','O','PAP_Cterminal_raw','PAP_Cterminal_calibrated','PAP_full_raw','PAP_full_calibrated']:
            s=b.scores(z.delta.to_numpy(),z[model].to_numpy());
            if model.endswith('_raw'):s['mae']=np.nan
            result.append(dict(family=fam,model=model,**s))
    d.to_parquet(out/'exploratory_additional_substitution_predictions.parquet',index=False);pd.DataFrame(result).to_csv(out/'exploratory_additional_substitution_metrics.csv',index=False)
    primary=pd.read_parquet(out/'mutation_predictions.parquet');coupling=[]
    for fam,z in primary.groupby('mutation_group'):
        coupling.append(dict(family=fam,n=len(z),baseline_vs_observed_delta_rho=b.rho(z.mean_stability_WT.to_numpy(),z.delta.to_numpy()),replicate_delta_agreement=b.rho((z.stability_r1_mut-z.stability_r1_WT).to_numpy(),(z.stability_r2_mut-z.stability_r2_WT).to_numpy())))
    pd.DataFrame(coupling).to_csv(out/'mutation_coupling_diagnostics.csv',index=False)
    b.dump('SECONDARY_DIAGNOSTICS_SCOPE.json',dict(status='complete',design='late exploratory, after primary outcomes; no primary model changes or validation model selection',additional_edit_rows=len(allpairs),specifically_verified=int(allpairs.specific_edit_verified.sum()),measured_rows=len(d),localization_strata='No verified experimental localization labels in the pinned assay inputs; no inferred labels invented.',top_prioritization='Descriptive top-decile selection on common rows, not prospective candidate validation; length and threshold confounding remain.'))
    print(json.dumps(dict(verified=len(d),families=d.mutation_group.value_counts().to_dict()),indent=2))
if __name__=='__main__':
    with b.threadpool_limits(limits=4):main()
