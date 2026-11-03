"""Independent source/score checks and explicitly descriptive secondary summaries."""
from pathlib import Path
import json,hashlib,importlib.metadata,re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
P=Path(__file__).resolve().parent;R=P/'results';rng=np.random.default_rng(20261031)
checks={};intervals=[];subgroups=[];thresholds=[]
def rho(y,p):
    return float(spearmanr(y,p).statistic) if len(y)>2 and np.std(p)>0 else np.nan
def ci(d,ycol,pcol,label,mae=False):
    d=d.dropna(subset=[ycol,pcol]);y=d[ycol].to_numpy(float);p=d[pcol].to_numpy(float);g=d.eval_group.to_numpy();ix=[np.flatnonzero(g==v) for v in np.unique(g)];vals=[]
    for _ in range(1000):
        k=np.concatenate([ix[j] for j in rng.integers(len(ix),size=len(ix))]);vals.append([rho(y[k],p[k]),float(np.mean(abs(y[k]-p[k])))])
    a=np.asarray(vals);intervals.append(dict(cohort=label,model=pcol,n=len(d),rho=rho(y,p),rho_low=float(np.nanquantile(a[:,0],.025)),rho_high=float(np.nanquantile(a[:,0],.975)),mae=float(np.mean(abs(y-p))) if mae else None,mae_low=float(np.nanquantile(a[:,1],.025)) if mae else None,mae_high=float(np.nanquantile(a[:,1],.975)) if mae else None))
def main():
    native=pd.read_parquet(R/'native_GFP_MP.parquet');G=pd.read_parquet(P/'sequence_dependence_groups.parquet').set_index('sequence')
    # Recompute all reported native metrics directly from the saved row predictions.
    m=pd.read_csv(R/'metrics.csv');maxdiff=0.
    for label in ['native_GFP_MP','native_MP_mCherry','native_GFP_MP_mutagenesis','sensitivity_identity40','sensitivity_seed32','sensitivity_seed33']:
        d=pd.read_parquet(R/(label+'.parquet'))
        for row in m[(m.cohort==label)&m.subset.eq('all')].itertuples():
            ok=d[row.model].notna();v=rho(d.loc[ok,'mean_stability'],d.loc[ok,row.model]);maxdiff=max(maxdiff,abs(v-row.rho))
        assert d.groupby('eval_group').fold.nunique().max()==1
    checks['metric_recompute_max_absolute_difference']=maxdiff;assert maxdiff<1e-12
    checks['native_group_fold_separation']=True
    raw=pd.read_excel(P/'inputs/sources/Guo2026_mmc4.xlsx',sheet_name='GFP_MP')
    checks['Guo_source_columns']=raw.columns.tolist()
    # Independent Yang replicate read, joined by construct name rather than row order.
    v=pd.read_parquet(R/'Yang_validation.parquet');rawmeans={}
    for sheet,n in [('Fig 4d',5),('Fig 4n',4)]:
        x=pd.read_excel(P/'inputs/sources/Yang2024_mmc12.xlsx',sheet_name=sheet,header=2)
        for row in x.loc[x.iloc[:,0].notna()].itertuples(index=False,name=None):rawmeans[row[0]]=float(np.mean(row[1:1+n]))
    err=max(abs(rawmeans[row.orf_name]-row.untreated_mean) for row in v.itertuples());assert err<1e-12;checks['Yang_independent_raw_replicate_mean_max_error']=err
    assert (v.fusion_sequence==v.native_sequence+'DYKDDDDK').all();checks['Yang_construct_sequence_FLAG_verified']=True
    a=pd.read_parquet(P/'natural_assay_records.parquet');guogroups=set(G.loc[a.protein_sequence,'primary_group']);assert not set(v.loc[v.strict_disjoint,'eval_group'])&guogroups
    checks['Yang_strict_disjoint_n']=int(v.strict_disjoint.sum());checks['Yang_exact_disjoint_n']=int((~v.exact_Guo_natural_sequence_overlap).sum())
    ref=pd.read_parquet(R/'annotation_training_membership.parquet');included=ref[ref.included];assert not set(included.component)&(guogroups|set(v.eval_group));assert included.groupby('pair').size().eq(2).all();checks['annotation_training_no_assay_component_overlap']=True
    # Common coverage avoids interpreting missing scores as algorithmic failures.
    common=[]
    e0=pd.read_parquet(R/'additional_transfer.parquet')
    for label,data,names in [('native',native,['L','A','T','O','AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop']),('additional_strict',e0[e0.strict_disjoint],['L','A','T','O','AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop','Guo_regression'])]:
        d=data.dropna(subset=names)
        for name in names:common.append(dict(cohort=label,model=name,n=len(d),rho=rho(d.mean_stability,d[name])))
    pd.DataFrame(common).to_csv(R/'common_coverage_metrics.csv',index=False)
    # Formula validation using independently parsed, safe literal assignments.
    text=(P/'inputs/Voutsinos2026_code/PAP.py').read_text();p=json.loads((P/'inputs/PAP_HUMAN30_PARSED_COEFFICIENTS.json').read_text());parsed={}
    for fam,motif,val in re.findall(r'human30\["(comp|pssm|pair|triplet)"\]\["([^"]+)"\]\s*=\s*([-+\d.eE]+)',text):parsed[(fam,motif)]=float(val)
    assert all(parsed[(fam,k)]==val for fam in ['comp','pssm','pair','triplet'] for k,val in p[fam].items() if val!=0)
    feat=pd.read_parquet(P/'sequence_features.parquet').set_index('sequence');err=0.
    for s in feat.index[::31]:
        ss=s[-30:];score=p['intercept']+30*sum(p['comp'][aa]*ss.count(aa) for aa in p['comp'])/len(ss)
        for family in ['pssm','pair','triplet']:
            for motif,val in p[family].items():
                matches=True
                for token in motif.split(':'):
                    pos=int(token[1:]);matches &= len(ss)>=pos and ss[-pos]==token[0]
                score+=val*matches
        err=max(err,abs(score-feat.loc[s,'PAP_Cterminal']))
    assert err<1e-12;checks['PAP_independent_coefficient_formula_max_error']=err
    # Descriptive uncertainty, conditional on these fitted predictions.
    for c in ['A','T','O','Shuffle0','AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop']:ci(native,'mean_stability',c,'native_GFP_MP',mae=c in ['A','T','O','Shuffle0'])
    e=pd.read_parquet(R/'additional_transfer.parquet');e=e[e.strict_disjoint]
    for c in ['A','T','O','AAC_resemblance','PAP_Cterminal','ShortStop','Guo_regression']:ci(e,'mean_stability',c,'additional_strict',mae=c in ['A','T','O','Guo_regression'])
    mut=pd.read_parquet(R/'mutation_predictions.parquet')
    assert (mut.loc[mut.mutation_group.eq('L_to_A'),'protein_sequence_mut'].to_numpy()==mut.loc[mut.mutation_group.eq('L_to_A'),'protein_sequence_WT'].str.replace('L','A',regex=False).to_numpy()).all()
    assert (mut.loc[mut.mutation_group.eq('microprotein_10mer_ext'),'protein_sequence_mut'].to_numpy()==(mut.loc[mut.mutation_group.eq('microprotein_10mer_ext'),'protein_sequence_WT']+'DNYNEPKANQ').to_numpy()).all();checks['all_2021_mutation_edits_independently_verified']=True
    for family,d in mut.groupby('mutation_group'):
        for c in ['T','O','PAP_Cterminal','PAP_Cterminal_calibrated','PAP_full_calibrated','delta_trained_T','cross_family_T']:ci(d,'delta',c,family,mae=c!='PAP_Cterminal')
    for c in ['published_PepScore','Length','PAP_Cterminal_fusion_sequence','PAP_full_fusion_sequence','MP_mCherry_T_fusion_sequence','MP_mCherry_O_fusion_sequence','GFP_MP_T_native_sequence','ShortStop']:ci(v[v.strict_disjoint],'untreated_mean',c,'Yang_strict')
    release=pd.read_parquet(P/'inputs/ShortStop_full_release_sequence_verified.parquet');assert len(release)==7152 and release.valid.all()
    values=release.groupby('sequence').mean_sam_probability.agg(['nunique','first']);values=values[values['nunique'].eq(1)]['first'];expected=v.native_sequence.map(values)
    assert np.allclose(v.ShortStop,expected,equal_nan=True)
    checks['ShortStop_release_rows_translated']=len(release);checks['Yang_ShortStop_all_coverage']=int(v.ShortStop.notna().sum());checks['Yang_ShortStop_strict_coverage']=int(v.loc[v.strict_disjoint,'ShortStop'].notna().sum())
    # Scope summaries: retain all subgroup results, with no p-value screening.
    native['low_complexity']=G.loc[native.protein_sequence,'low_complexity'].to_numpy()
    masks={'all':np.ones(len(native),bool),'length_le30':native.sequence_length.le(30),'length_ge31':native.sequence_length.ge(31),'both_replicates':native.stability_r1.notna()&native.stability_r2.notna(),'low_complexity':native.low_complexity,'other_complexity':~native.low_complexity}
    for context in native.source_biotype.dropna().unique():masks['context_'+context]=native.source_biotype.eq(context)
    for name,mask in masks.items():
        d=native[mask]
        for c in ['A','T','O','AAC_resemblance','PAP_Cterminal','ShortStop']:
            z=d.dropna(subset=[c]);subgroups.append(dict(subset=name,model=c,n=len(z),rho=rho(z.mean_stability,z[c])))
    # Threshold calls and continuous ranking are different estimands. Length-stratified
    # permutation is a diagnostic only: no observational result establishes function.
    d=native[native.ShortStop.notna()].copy();d['length_bin']=pd.qcut(d.sequence_length,10,duplicates='drop')
    for flag in ['sam','sam_consensus_90','sam_consensus_100']:
        ispos=d[flag].eq(True);thresholds.append(dict(threshold=flag,n_called=int(ispos.sum()),n_other=int((~ispos).sum()),median_score_called=float(d.loc[ispos,'mean_stability'].median()),median_score_other=float(d.loc[~ispos,'mean_stability'].median()),median_length_called=float(d.loc[ispos,'sequence_length'].median()),median_length_other=float(d.loc[~ispos,'sequence_length'].median())))
    yy=d.mean_stability.to_numpy();ss=d.ShortStop.to_numpy();l=np.log(d.sequence_length.to_numpy());z=np.column_stack([np.ones(len(d)),l,l*l]);ry=yy-z@np.linalg.lstsq(z,yy,rcond=None)[0];rs=ss-z@np.linalg.lstsq(z,ss,rcond=None)[0]
    checks['ShortStop_length_residual_rank_association_descriptive']=rho(ry,rs)
    checks['native_complete_replicates']=int((native.stability_r1.notna()&native.stability_r2.notna()).sum());checks['native_le30']=int(native.sequence_length.le(30).sum())
    # All three annotation seeds, no outcome-based winner selection.
    seedrows=[]
    for seed in [100,101,102]:
        s=pd.read_parquet(R/f'annotation_seed{seed}.parquet').set_index('sequence').score
        seedrows.append(dict(seed=seed,native_rho=rho(native.mean_stability,s.loc[native.protein_sequence].to_numpy()),additional_strict_rho=rho(e.mean_stability,s.loc[e.protein_sequence].to_numpy())))
    pd.DataFrame(seedrows).to_csv(R/'annotation_seed_sensitivity.csv',index=False)
    pd.DataFrame(intervals).to_csv(R/'conditional_intervals.csv',index=False);pd.DataFrame(subgroups).to_csv(R/'subgroup_summaries.csv',index=False);pd.DataFrame(thresholds).to_csv(R/'ShortStop_threshold_summaries.csv',index=False)
    checks['status']='passed';checks['uncertainty']='1000 operational-group bootstrap draws conditional on fitted predictions; secondary descriptive intervals, no multiplicity-adjusted claims';(R/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n')
    versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','matplotlib','pyarrow','openpyxl','threadpoolctl','pymupdf']};(P/'environment_versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
