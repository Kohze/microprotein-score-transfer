"""Retrospective grouped benchmark. No validation labels enter model fitting."""
from pathlib import Path
import sys,os,json,time,hashlib,warnings
ROOT=Path(__file__).resolve().parent; WS=ROOT.parent
sys.path.insert(0,str(WS/'thesis_reanalysis_2026-10-31/vendor'))
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',message='An input array is constant')
ALPHAS=np.array([.1,1,10,100,1000]);SEED=20261031
F=pd.read_parquet(ROOT/'sequence_features.parquet').set_index('sequence')
SETS=json.loads((ROOT/'FEATURE_DEFINITION.json').read_text())['feature_sets']
G=pd.read_parquet(ROOT/'sequence_dependence_groups.parquet').set_index('sequence')
A=pd.read_parquet(ROOT/'natural_assay_records.parquet')
M=pd.read_parquet(ROOT/'primary_mutation_pairs.parquet')
Y=pd.read_parquet(ROOT/'validation_assay_records.parquet')
SS=pd.read_parquet(ROOT/'inputs/ShortStop_Guo_sequence_checked_mapping.parquet')
SS=SS[SS.exact_reference_sequence & SS.exact_shortstop_sequence].drop_duplicates(['source_sheet','group','protein_identifier','protein_sequence'])
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
receipts=[];calibrations=[];metrics=[]

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def groups(seq,kind='primary_group'):return G.loc[list(seq),kind].to_numpy()
def folds(g,n=5,seed=SEED):
    u,c=np.unique(g,return_counts=True);p=np.random.default_rng(seed).permutation(len(u));p=p[np.argsort(-c[p],kind='stable')]
    sizes=np.zeros(n,int);mapping={}
    for j in p:
        k=int(sizes.argmin());mapping[u[j]]=k;sizes[k]+=c[j]
    return np.array([mapping[v] for v in g])
def rho(y,p):
    ok=np.isfinite(y)&np.isfinite(p)
    return float(spearmanr(np.asarray(y)[ok],np.asarray(p)[ok]).statistic) if ok.sum()>2 and np.std(np.asarray(p)[ok])>0 else np.nan
def scores(y,p):
    ok=np.isfinite(y)&np.isfinite(p);y=np.asarray(y)[ok];p=np.asarray(p)[ok]
    return dict(n=int(len(y)),rho=rho(y,p),mae=float(np.mean(abs(y-p))) if len(y) else np.nan)
def addmetric(cohort,model,y,p,subset='all',**extra):
    s=scores(y,p)
    if extra.get('scale')=='raw' or extra.get('mae_comparable') is False:s['mae']=np.nan
    metrics.append(dict(cohort=cohort,model=model,subset=subset,**s,**extra))
def X(seq,name):return F.loc[list(seq),SETS[name]].to_numpy(float)
def ridge_path(x,y,z,alphas=ALPHAS):
    mean=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.; xx=(x-mean)/sd;zz=(z-mean)/sd
    ymean=float(y.mean());v,q=eigh(xx.T@xx,check_finite=False);v=np.maximum(v,0)
    projected=q.T@(xx.T@(y-ymean));coefs=q@(projected[:,None]/(v[:,None]+np.asarray(alphas)[None,:]))
    return zz@coefs+ymean,(mean,sd,coefs,ymean)
def fit_predict(x,y,g,z,seed=SEED):
    inner=folds(g,3,seed);err=np.zeros(len(ALPHAS));total=0
    for k in range(3):
        tr=inner!=k;te=~tr;assert not set(g[tr])&set(g[te])
        pred,_=ridge_path(x[tr],y[tr],x[te]);err+=abs(y[te,None]-pred).sum(0);total+=te.sum()
    best=int(np.argmin(err));pred,_=ridge_path(x,y,z,[ALPHAS[best]])
    return pred[:,0],float(ALPHAS[best]),(err/total).tolist()
def nested(seq,y,g,name,seed=SEED,zseq=None,trees=False,label=''):
    x=X(seq,name);z=x if zseq is None else X(zseq,name); ff=folds(g,5,seed);p=np.full(len(y),np.nan);q=np.full(len(y),np.nan)
    for k in range(5):
        tr=ff!=k;te=~tr;assert not set(g[tr])&set(g[te])
        if trees:
            mod=ExtraTreesRegressor(n_estimators=200,max_depth=10,min_samples_leaf=10,max_features=.7,random_state=100,n_jobs=4).fit(x[tr],y[tr]);p[te]=mod.predict(x[te]);q[te]=mod.predict(z[te]);alpha=None;err=[]
        else:
            pp,alpha,err=fit_predict(x[tr],y[tr],g[tr],np.vstack([x[te],z[te]]),seed+k+19);p[te]=pp[:te.sum()];q[te]=pp[te.sum():]
        receipts.append(dict(cohort=label,model=name+('_ET' if trees else ''),seed=seed,fold=k,n_train=int(tr.sum()),n_test=int(te.sum()),alpha=alpha,inner_mae=err,overlapping_groups=0))
    return p,q,ff
def calibrated(s,y,g,s_mut=None,label=''):
    ff=folds(g);p=np.full(len(y),np.nan);q=np.full(len(y),np.nan)
    for k in range(5):
        tr=(ff!=k)&np.isfinite(s);te=(ff==k)&np.isfinite(s)
        increasing=rho(y[tr],s[tr])>=0
        if tr.sum()<3 or not te.any():continue
        mod=IsotonicRegression(increasing=increasing,out_of_bounds='clip').fit(s[tr],y[tr]);p[te]=mod.predict(s[te]);
        if s_mut is not None:
            ok=(ff==k)&np.isfinite(s_mut);q[ok]=mod.predict(s_mut[ok])
        calibrations.append(dict(cohort=label,fold=k,increasing=bool(increasing),n_train=int(tr.sum())))
    return p,q
def fixed(d,annotation):
    s=d.protein_sequence.tolist();out=pd.DataFrame(index=np.arange(len(d)))
    out['AAC_resemblance']=annotation.predict_proba(F.loc[s,[f'AAC_{a}' for a in 'ACDEFGHIKLMNPQRSTVWY']])[:,1]
    out['PAP_Cterminal']=F.loc[s,'PAP_Cterminal'].to_numpy();out['PAP_full']=F.loc[s,'PAP_full'].to_numpy()
    keys=['source_sheet','group','protein_identifier','protein_sequence']
    joined=d[keys].merge(SS[keys+['mean_sam_probability','sam','sam_consensus_90','sam_consensus_100']],on=keys,how='left',validate='one_to_one')
    out['ShortStop']=joined.mean_sam_probability.to_numpy()
    for c in ['sam','sam_consensus_90','sam_consensus_100']:out[c]=joined[c].to_numpy()
    return out
def benchmark_native(d,label,ann,full=True,kind='primary_group',seed=SEED):
    d=d[d.mean_stability.notna() & ~d.inconsistent_parent_id].copy().reset_index(drop=True)
    y=d.mean_stability.to_numpy(float);g=groups(d.protein_sequence,kind);out=d.copy();out['eval_group']=g;out['fold']=folds(g,seed=seed)
    models=list(SETS) if full else ['L','A','T','O','Shuffle0']
    for name in models:
        p,_,_=nested(d.protein_sequence,y,g,name,seed,label=label);out[name]=p;addmetric(label,name,y,p)
    if full:
        for name in ['A','T','O']:
            p,_,_=nested(d.protein_sequence,y,g,name,seed,trees=True,label=label);out[name+'_ET']=p;addmetric(label,name+'_ET',y,p)
    fs=fixed(d,ann)
    for name in ['AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop']:
        s=fs[name].to_numpy(float);p,_=calibrated(s,y,g,label=label+'::'+name);out[name]=s;out[name+'_calibrated']=p;addmetric(label,name,y,s,scale='raw');addmetric(label,name+'_calibrated',y,p,scale='WT_calibrated')
    for c in ['sam','sam_consensus_90','sam_consensus_100']:out[c]=fs[c].to_numpy()
    out.to_parquet(OUT/(label+'.parquet'),index=False)
    print('Completed',label,len(d),flush=True);return out
def annotation_classifier():
    ref=pd.read_parquet(ROOT/'annotation_reference.parquet');ref['component']=groups(ref.sequence)
    blocked=set(groups(A.protein_sequence))|set(groups(Y.native_sequence));bad=set(ref.loc[ref.component.isin(blocked),'pair'])
    # Propagate pair and reference split-group exclusions to fixed point.
    keep=~ref.pair.isin(bad)
    for _ in range(len(ref)):
        badsplit=set(ref.loc[~keep,'split_group']);badcomp=set(ref.loc[~keep,'component']);badpairs=set(ref.loc[ref.split_group.isin(badsplit)|ref.component.isin(badcomp),'pair']);new=~ref.pair.isin(badpairs|bad)
        if np.array_equal(new,keep):break
        keep=new
    r=ref[keep];assert r.groupby('pair').size().eq(2).all();assert not set(r.component)&blocked
    ref['included']=keep;ref.to_parquet(OUT/'annotation_training_membership.parquet',index=False)
    cols=[f'AAC_{a}' for a in 'ACDEFGHIKLMNPQRSTVWY'];model=None
    for seed in [100,101,102]:
        m=ExtraTreesClassifier(n_estimators=200,max_depth=10,min_samples_leaf=10,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=4).fit(F.loc[r.sequence,cols],r['class'].eq('canonical').astype(int))
        if seed==100:model=m
        pd.DataFrame({'sequence':F.index,'score':m.predict_proba(F[cols])[:,1]}).to_parquet(OUT/f'annotation_seed{seed}.parquet',index=False)
    dump('annotation_audit.json',dict(rows=int(keep.sum()),pairs=int(keep.sum()/2),removed_pairs=int((~keep).sum()/2),blocked_components=len(blocked),class_counts=r['class'].value_counts().to_dict()))
    assert len(model.classes_)==2 and model.classes_.tolist()==[0,1]
    return model
def bootstrap_contrast(d,label,b=1000):
    yy=d.mean_stability.to_numpy();t=d['T'].to_numpy();o=d['O'].to_numpy();g=d.eval_group.to_numpy()
    inds=[np.flatnonzero(g==u) for u in np.unique(g)];rng=np.random.default_rng(SEED);vals=[]
    for _ in range(b):
        ix=np.concatenate([inds[j] for j in rng.integers(len(inds),size=len(inds))]);vals.append([rho(yy[ix],o[ix])-rho(yy[ix],t[ix]),np.mean(abs(yy[ix]-t[ix]))-np.mean(abs(yy[ix]-o[ix]))])
    arr=np.asarray(vals);return dict(cohort=label,n=len(d),rank_gain=rho(yy,o)-rho(yy,t),mae_reduction=float(np.mean(abs(yy-t))-np.mean(abs(yy-o))),rank_gain_CI=np.quantile(arr[:,0],[.025,.975]).tolist(),mae_reduction_CI=np.quantile(arr[:,1],[.025,.975]).tolist(),draws=b)
def extra_transfer(dev,ann):
    extra=A[A.group.isin(['microprotein_ms_HLA','microprotein_ms_nonHLA']) & A.mean_stability.notna()].copy().reset_index(drop=True);extra['eval_group']=groups(extra.protein_sequence);extra['strict_disjoint']=~extra.eval_group.isin(set(dev.eval_group));y=dev.mean_stability.to_numpy();g=dev.eval_group.to_numpy()
    for name in ['L','A','T','O','Shuffle0']:
        extra[name],alpha,err=fit_predict(X(dev.protein_sequence,name),y,g,X(extra.protein_sequence,name));receipts.append(dict(cohort='additional_transfer',model=name,alpha=alpha,inner_mae=err))
    fs=fixed(extra,ann); ds=fixed(dev,ann)
    for name in ['AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop']:
        extra[name]=fs[name].to_numpy();s=ds[name].to_numpy(float);ok=np.isfinite(s);inc=rho(y[ok],s[ok])>=0;m=IsotonicRegression(increasing=inc,out_of_bounds='clip').fit(s[ok],y[ok]);extra[name+'_calibrated']=np.nan;ok2=np.isfinite(extra[name]);extra.loc[ok2,name+'_calibrated']=m.predict(extra.loc[ok2,name])
    pub=pd.read_excel(ROOT/'inputs/sources/Guo2026_mmc7.xlsx',sheet_name='Predicted_stablity').rename(columns={'Predicted stability [Classification]':'Guo_classification','Predicted stability [Regression]':'Guo_regression'})
    assert not pub.protein_sequence.duplicated().any()
    extra=extra.merge(pub[['protein_sequence','Guo_classification','Guo_regression']],on='protein_sequence',how='left',validate='many_to_one')
    for subset,mask in [('all',np.ones(len(extra),bool)),('strict_disjoint',extra.strict_disjoint.to_numpy())]:
        for name in ['L','A','T','O','Shuffle0','AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop','Guo_classification','Guo_regression','AAC_resemblance_calibrated','PAP_Cterminal_calibrated','PAP_full_calibrated','ShortStop_calibrated']:
            raw=name in ['AAC_resemblance','PAP_Cterminal','PAP_full','ShortStop','Guo_classification']
            addmetric('additional_transfer',name,extra.loc[mask,'mean_stability'].to_numpy(),extra.loc[mask,name].to_numpy(float),subset,scale='raw' if raw else 'reporter')
    extra.to_parquet(OUT/'additional_transfer.parquet',index=False);return extra
def mutations(ann):
    wt=A[(A.source_sheet=='GFP_MP_mutagenesis')&(A.group=='microprotein_WT')&A.mean_stability.notna()].copy().reset_index(drop=True);g=groups(wt.protein_sequence);yy=wt.mean_stability.to_numpy();ff=folds(g)
    pairs=M[~M.inconsistent_parent_id & M.mean_stability_mut.notna() & M.mean_stability_WT.notna()].copy().reset_index(drop=True);pairs['eval_group']=groups(pairs.protein_sequence_WT);pairs['delta']=pairs.mean_stability_mut-pairs.mean_stability_WT
    pairfold=dict(zip(g,ff));pairs['fold']=pairs.eval_group.map(pairfold);assert pairs.fold.notna().all()
    for name in ['L','A','T','O','Shuffle0']:
        pairs[name]=np.nan;x=X(wt.protein_sequence,name)
        for k in range(5):
            tr=ff!=k;te=pairs.fold.eq(k).to_numpy();z=np.vstack([X(pairs.loc[te,'protein_sequence_WT'],name),X(pairs.loc[te,'protein_sequence_mut'],name)]);p,alpha,err=fit_predict(x[tr],yy[tr],g[tr],z,SEED+k+19);n=te.sum();pairs.loc[te,name]=p[n:]-p[:n]
    cols=[f'AAC_{a}' for a in 'ACDEFGHIKLMNPQRSTVWY']
    for name in ['AAC_resemblance','PAP_Cterminal','PAP_full']:
        def value(seq):return ann.predict_proba(F.loc[list(seq),cols])[:,1] if name=='AAC_resemblance' else F.loc[list(seq),name].to_numpy()
        sw=value(wt.protein_sequence);pw=value(pairs.protein_sequence_WT);pm=value(pairs.protein_sequence_mut);pairs[name]=pm-pw;pairs[name+'_calibrated']=np.nan
        for k in range(5):
            tr=ff!=k;te=pairs.fold.eq(k).to_numpy();inc=rho(yy[tr],sw[tr])>=0;mod=IsotonicRegression(increasing=inc,out_of_bounds='clip').fit(sw[tr],yy[tr]);pairs.loc[te,name+'_calibrated']=mod.predict(pm[te])-mod.predict(pw[te]);calibrations.append(dict(cohort='mutation::'+name,fold=k,increasing=bool(inc),n_train=int(tr.sum())))
    pairs['zero']=0.;pairs['family_mean']=np.nan;pairs['cross_family_mean']=np.nan
    for family,df in pairs.groupby('mutation_group'):
        ix=df.index.to_numpy();gy=df.eval_group.to_numpy();dy=df.delta.to_numpy();fold=folds(gy)
        for name in ['A','T','O']:
            xx=X(df.protein_sequence_mut,name)-X(df.protein_sequence_WT,name);pred=np.full(len(df),np.nan);cross=pred.copy()
            for k in range(5):
                tr=fold!=k;te=~tr;pred[te],_,_=fit_predict(xx[tr],dy[tr],gy[tr],xx[te],SEED+k+19)
                other=pairs[(pairs.mutation_group!=family)&~pairs.eval_group.isin(set(gy[te]))];ox=X(other.protein_sequence_mut,name)-X(other.protein_sequence_WT,name);cross[te],_,_=fit_predict(ox,other.delta.to_numpy(),other.eval_group.to_numpy(),xx[te],SEED+k+19)
                pairs.loc[ix[te],'family_mean']=float(dy[tr].mean());pairs.loc[ix[te],'cross_family_mean']=float(other.delta.mean())
            pairs.loc[ix,'delta_trained_'+name]=pred;pairs.loc[ix,'cross_family_'+name]=cross
        for name in ['zero','family_mean','L','A','T','O','Shuffle0','AAC_resemblance','AAC_resemblance_calibrated','PAP_Cterminal','PAP_Cterminal_calibrated','PAP_full','PAP_full_calibrated','delta_trained_A','delta_trained_T','delta_trained_O','cross_family_A','cross_family_T','cross_family_O','cross_family_mean']:
            addmetric('mutation_'+family,name,dy,pairs.loc[ix,name].to_numpy(float))
        for rep in [1,2]:
            ry=(df[f'stability_r{rep}_mut']-df[f'stability_r{rep}_WT']).to_numpy(float)
            for name in ['A','T','O','PAP_Cterminal_calibrated','PAP_full_calibrated']:
                addmetric('mutation_'+family,name,ry,pairs.loc[ix,name].to_numpy(float),subset=f'replicate{rep}')
    pairs.to_parquet(OUT/'mutation_predictions.parquet',index=False);return pairs
def validate(ann):
    # Main model selection and analysis are persisted before these outcomes are read.
    dump('VALIDATION_RELEASE.json',dict(utc=pd.Timestamp.now(tz='UTC').isoformat(),main_prediction_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.parquet')},no_validation_fitting=True,primary_orientation='MP_mCherry',primary_representation='fusion_sequence'))
    out=Y.copy();out['eval_group']=groups(out.native_sequence);allgroups=set(groups(A.protein_sequence));out['strict_disjoint']=~out.eval_group.isin(allgroups)
    cols=[f'AAC_{a}' for a in 'ACDEFGHIKLMNPQRSTVWY']
    for rep in ['native_sequence','fusion_sequence']:
        for name in ['PAP_Cterminal','PAP_full']:out[name+'_'+rep]=F.loc[list(out[rep]),name].to_numpy()
        out['AAC_resemblance_'+rep]=ann.predict_proba(F.loc[list(out[rep]),cols])[:,1]
        for sheet in ['MP_mCherry','GFP_MP']:
            tr=A[(A.source_sheet==sheet)&(A.group=='microprotein_WT')&A.mean_stability.notna()];g=groups(tr.protein_sequence);y=tr.mean_stability.to_numpy()
            for name in ['L','A','T','O','Shuffle0']:
                # Exclude every validation-connected training group, including the ten exact overlaps.
                keep=~np.isin(g,out.eval_group);p,alpha,err=fit_predict(X(tr.loc[keep,'protein_sequence'],name),y[keep],g[keep],X(out[rep],name));key=f'{sheet}_{name}_{rep}';out[key]=p;receipts.append(dict(cohort='Yang_transfer',model=key,n_train=int(keep.sum()),alpha=alpha,inner_mae=err))
    # A frozen score is used only where exact native peptide uniquely determines it.
    release=pd.read_parquet(ROOT/'inputs/ShortStop_full_release_sequence_verified.parquet')
    native_ss=release.loc[release.valid,['sequence','mean_sam_probability']].drop_duplicates().rename(columns={'sequence':'protein_sequence'});counts=native_ss.groupby('protein_sequence').mean_sam_probability.nunique();unique=native_ss[native_ss.protein_sequence.isin(counts[counts.eq(1)].index)].drop_duplicates('protein_sequence').set_index('protein_sequence').mean_sam_probability
    out['ShortStop']=out.native_sequence.map(unique);out['Length']=out.native_length
    outcome=out.untreated_mean.to_numpy();positive=outcome>.05
    predictor=['published_PepScore','ShortStop','Length']+[c for c in out if c.startswith(('PAP_','AAC_','MP_mCherry_','GFP_MP_'))]
    for subset,mask in [('all',np.ones(len(out),bool)),('exact_disjoint',~out.exact_Guo_natural_sequence_overlap.to_numpy()),('strict_disjoint',out.strict_disjoint.to_numpy())]:
        for name in predictor:
            p=out[name].to_numpy(float);ok=mask&np.isfinite(p);auc=float(roc_auc_score(positive[ok],p[ok])) if len(np.unique(positive[ok]))==2 else np.nan
            addmetric('Yang_validation',name,outcome[ok],p[ok],subset,auroc=auc,positives=int(positive[ok].sum()),mae_comparable=False)
    out.to_parquet(OUT/'Yang_validation.parquet',index=False);return out
def main():
    start=time.time();freeze=json.loads((ROOT/'INPUT_FREEZE.json').read_text());assert freeze['protocol_sha256']==hashlib.sha256((ROOT/'EXECUTION_PROTOCOL.txt').read_bytes()).hexdigest()
    # Independent check of eigensolver ridge predictions against sklearn.
    xx=X(A.protein_sequence.iloc[:100],'O');yy=A.mean_stability.iloc[:100].to_numpy();p,state=ridge_path(xx[:70],yy[:70],xx[70:],[10]);mean,sd,coef,ym=state;check=Ridge(alpha=10,solver='cholesky').fit((xx[:70]-mean)/sd,yy[:70]).predict((xx[70:]-mean)/sd);err=float(np.max(abs(p[:,0]-check)));assert err<1e-9;dump('ridge_numerical_check.json',dict(max_absolute_difference=err,tolerance=1e-9))
    ann=annotation_classifier();dev=benchmark_native(A[(A.source_sheet=='GFP_MP')&(A.group=='microprotein_WT')],'native_GFP_MP',ann)
    contrasts=[bootstrap_contrast(dev,'native_GFP_MP')]
    for sheet in ['MP_mCherry','GFP_MP_mutagenesis']:
        d=benchmark_native(A[(A.source_sheet==sheet)&(A.group=='microprotein_WT')],'native_'+sheet,ann,full=False);contrasts.append(bootstrap_contrast(d,'native_'+sheet))
    for kind,seed,label in [('sensitivity_group',SEED,'sensitivity_identity40'),('primary_group',SEED+1,'sensitivity_seed32'),('primary_group',SEED+2,'sensitivity_seed33')]:
        d=benchmark_native(A[(A.source_sheet=='GFP_MP')&(A.group=='microprotein_WT')],label,ann,full=False,kind=kind,seed=seed);contrasts.append(bootstrap_contrast(d,label))
    extra=extra_transfer(dev,ann);mut=mutations(ann)
    pd.DataFrame(metrics).to_csv(OUT/'metrics_before_validation.csv',index=False);dump('native_contrasts.json',contrasts)
    val=validate(ann)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);dump('fit_receipts.json',receipts);dump('calibration_directions.json',calibrations)
    dump('RUN_COMPLETE.json',dict(seconds=time.time()-start,protocol_sha256=freeze['protocol_sha256'],main_rows=len(dev),additional_rows=len(extra),additional_strict=int(extra.strict_disjoint.sum()),mutation_pairs=len(mut),Yang_rows=len(val),Yang_strict=int(val.strict_disjoint.sum()),no_validation_outcome_model_selection=True,bootstrap='conditional on fitted predictions',status='complete'))
    print((OUT/'RUN_COMPLETE.json').read_text(),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
