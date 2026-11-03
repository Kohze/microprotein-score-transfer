"""Short-valid, terminal-aware features and composition-conditioned order controls."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
AA='ACDEFGHIKLMNPQRSTVWY'
PROPS=['BHAR880101','BIGC670101','CHAM810101','CHAM820101','CHAM820102','CHOC760101','CIDH920105','DAYM780201']
HYDRO=dict(zip('ACDEFGHIKLMNPQRSTVWY',[1.8,2.5,-3.5,-3.5,2.8,-.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-.8,-.7,4.2,-.9,-1.3]))


def pap_parts(seq,param,window=False):
    s=seq[-30:] if window else seq
    comp=sum(param['comp'][a] for a in s)*30/len(s)
    term=0.
    for family in ['pssm','pair','triplet']:
        for motif,coef in param[family].items():
            if all(len(s)>=int(t[1:]) and s[-int(t[1:])]==t[0] for t in motif.split(':')):term+=coef
    return float(comp),float(term),float(param['intercept']+comp+term)


def main():
    assays=pd.read_parquet(ROOT/'inputs/Guo2026_all_assay_constructs.parquet')
    yang=pd.read_parquet(ROOT/'validation_assay_records.parquet')
    refs=pd.read_parquet(ROOT/'annotation_reference.parquet')
    sequences=set(assays.loc[assays.sequence_valid,'protein_sequence'])|set(yang.native_sequence)|set(yang.fusion_sequence)|set(refs.sequence)
    sequences=sorted(sequences)
    sha=hashlib.sha256('\n'.join(sequences).encode()).hexdigest()
    if (ROOT/'FEATURE_DEFINITION.json').exists() and (ROOT/'sequence_features.parquet').exists():
        old=json.loads((ROOT/'FEATURE_DEFINITION.json').read_text())
        if old['sequence_list_sha256']==sha:
            print('Using complete features for identical sequence list',len(sequences));return
    idx=pd.read_csv(ROOT/'inputs/reference/sources/AAidx_protr_1.7.4.csv').set_index('AccNo')
    matrix=idx.loc[PROPS,list(AA)].to_numpy(float).T
    matrix=(matrix-matrix.mean(axis=0))/matrix.std(axis=0,ddof=1)
    assert np.isfinite(matrix).all()
    param=json.loads((ROOT/'inputs/PAP_HUMAN30_PARSED_COEFFICIENTS.json').read_text())
    records=[];null_sds=[];started=time.time()
    for i,s in enumerate(sequences):
        assert len(s)>=6 and not set(s)-set(AA)
        a=np.array([AA.index(c) for c in s]);L=len(s)
        record={'sequence':s,'length':L,'log_length':float(np.log(L)),'log_length_sq':float(np.log(L)**2)}
        record.update({'AAC_'+c:s.count(c)/L for c in AA})
        for p in range(5):
            record.update({f'N{p+1}_{c}':float(s[p]==c) for c in AA})
            record.update({f'C{p+1}_{c}':float(s[-p-1]==c) for c in AA})
        for name,t in [('hydropathy',s),('N10_hydropathy',s[:10]),('C10_hydropathy',s[-10:])]:
            record[name]=float(np.mean([HYDRO[c] for c in t]))
        comp,term,pap=pap_parts(s,param,False)
        record.update(PAP_composition=comp,PAP_terminal=term,PAP_full=pap,PAP_Cterminal=pap_parts(s,param,True)[2])
        v=matrix[a]
        raw=np.array([(v[:-lag]*v[lag:]).mean(axis=0) for lag in range(1,6)])
        # Fixed first/last five residues protect native terminal motifs. The
        # 199 null draws and three controls are reproducible from sequence alone.
        seed=int(hashlib.sha256(('order_null_v1:'+s).encode()).hexdigest()[:16],16)
        rng=np.random.default_rng(seed)
        shuffled=np.tile(a,(202,1))
        if L>10:
            order=np.argsort(rng.random((202,L-10)),axis=1)
            shuffled[:,5:-5]=a[5:-5][order]
        sv=matrix[shuffled]
        vals=np.stack([(sv[:,:-lag]*sv[:,lag:]).mean(axis=1) for lag in range(1,6)],axis=1)
        mean=vals[:199].mean(axis=0);sd=vals[:199].std(axis=0,ddof=1)
        residual=np.divide(raw-mean,sd,out=np.zeros_like(raw),where=sd>1e-12)
        control=np.divide(vals[199:]-mean,sd,out=np.zeros_like(vals[199:]),where=sd>1e-12)
        assert np.isfinite(residual).all()
        for lag in range(1,6):
            for j,prop in enumerate(PROPS):
                key=f'{prop}_lag{lag}'
                record['order_z_'+key]=float(residual[lag-1,j])
                record['order_raw_'+key]=float(raw[lag-1,j])
                for rep in range(3):record[f'control{rep}_z_'+key]=float(control[rep,lag-1,j])
        null_sds.append(sd.flatten());records.append(record)
        if (i+1)%500==0:print('Feature sequences',i+1,'/',len(sequences),'seconds',round(time.time()-started,1),flush=True)
    frame=pd.DataFrame(records);frame.to_parquet(ROOT/'sequence_features.parquet',index=False)
    np.save(ROOT/'order_null_standard_deviations.npy',np.array(null_sds))
    length=['log_length','log_length_sq'];aac=[f'AAC_{c}' for c in AA]
    terminals=[c for c in frame if c.startswith(tuple(f'{end}{p}_' for end in ['N','C'] for p in range(1,6)))]
    terminal=length+aac+terminals+['hydropathy','N10_hydropathy','C10_hydropathy','PAP_composition','PAP_terminal']
    spec=dict(sequence_list_sha256=sha,rows=len(frame),properties=PROPS,property_standardization='mean/sd across 20 amino acids; ddof=1',
        lags=[1,2,3,4,5],null_draws=199,controls=3,preserved_terminals=5,
        feature_sets={'L':length,'A':length+aac,'T':terminal,
            'O':terminal+[c for c in frame if c.startswith('order_z_')],
            'RawOrder':terminal+[c for c in frame if c.startswith('order_raw_')]},
        zero_null_variance='zero residual if sd<=1e-12; tracked explicitly',
        independent_validation_outcomes_used=False,
        minimum_length=int(frame.length.min()),maximum_length=int(frame.length.max()),
        null_zero_sd_cells=int((np.array(null_sds)<=1e-12).sum()))
    for rep in range(3):spec['feature_sets'][f'Shuffle{rep}']=terminal+[c for c in frame if c.startswith(f'control{rep}_z_')]
    (ROOT/'FEATURE_DEFINITION.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({k:v for k,v in spec.items() if k!='feature_sets'},indent=2),flush=True)


if __name__=='__main__':main()
