"""Exhaustive eligible-pair similarity audit and operational dependence groups."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
import parasail
AA='ACDEFGHIKLMNPQRSTVWY'


class Union:
    def __init__(self,n):self.parent=list(range(n))
    def root(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i
    def join(self,i,j):
        a,b=self.root(i),self.root(j)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def main():
    natural=pd.read_parquet(ROOT/'natural_assay_records.parquet')
    yang=pd.read_parquet(ROOT/'validation_assay_records.parquet')
    reference=pd.read_parquet(ROOT/'annotation_reference.parquet')
    positives=pd.read_parquet(ROOT/'available_shortstop_positive_counterparts.parquet')
    sources=defaultdict(set);tokens=defaultdict(set)
    def add(seq,source,genes='',groups='',parent=''):
        sources[seq].add(source)
        for g in str(genes).split(';'):
            if g and g!='nan':tokens[seq].add('gene:'+g)
        for g in str(groups).split(';'):
            if g and g!='nan':tokens[seq].add('catalogue:'+g)
        if parent:tokens[seq].add('parent:'+parent)
    for r in natural.itertuples():add(r.protein_sequence,'Guo',r.genes,r.catalogue_groups,r.protein_identifier)
    for r in yang.itertuples():add(r.native_sequence,'Yang',r.genes,r.catalogue_groups,'Yang:'+r.orf_name)
    for r in reference.itertuples():add(r.sequence,'annotation_reference',r.genes,str(r.group))
    for r in positives.itertuples():add(r.sequence,'ShortStop_positive',parent='positive:'+r.id)
    sequences=sorted(sources);index={s:i for i,s in enumerate(sequences)}
    lengths=np.array(list(map(len,sequences)));counts=np.array([[s.count(a) for a in AA] for s in sequences],dtype=np.int16)
    base=Union(len(sequences));bytoken=defaultdict(list)
    for s in sequences:
        for t in tokens[s]:bytoken[t].append(index[s])
    for ids in bytoken.values():
        for i in ids[1:]:base.join(ids[0],i)
    # Exact identities among engineered constructs connect their parents even
    # if those parents have different native sequences.
    assays=pd.read_parquet(ROOT/'inputs/Guo2026_all_assay_constructs.parquet')
    parent_seq=natural.groupby('protein_identifier').protein_sequence.apply(lambda x:set(x)).to_dict()
    byconstruct=defaultdict(set)
    for r in assays.loc[assays.sequence_valid].itertuples():
        byconstruct[r.protein_sequence].update(index[s] for s in parent_seq.get(r.protein_identifier,set()) if s in index)
    for ids in byconstruct.values():
        ids=sorted(ids)
        for i in ids[1:]:base.join(ids[0],i)
    base_parent=base.parent.copy()
    dest=ROOT/'similarity_edges.parquet'
    started=time.time()
    if dest.exists():
        edges=pd.read_parquet(dest)
        assert json.loads((ROOT/'similarity_sequence_manifest.json').read_text())['sequences_sha256']==hashlib.sha256('\n'.join(sequences).encode()).hexdigest()
        print('Reusing completed exact-input similarity audit',len(edges),flush=True)
    else:
        records=[];scanned=0;eligible=0
        def scan(i):
            # Both sequences must have >=80% coverage. The count-overlap bound
            # cannot reject a pair satisfying 40% exact identity at that coverage.
            cand=np.flatnonzero((np.arange(len(sequences))>i)&(lengths>=.8*lengths[i])&(lengths*.8<=lengths[i]))
            if len(cand):
                bound=np.minimum(counts[i],counts[cand]).sum(axis=1)
                cand=cand[bound+1e-9>=.32*np.maximum(lengths[i],lengths[cand])]
            out=[]
            for j in cand:
                aln=parasail.sw_trace_striped_16(sequences[i],sequences[j],10,1,parasail.blosum62)
                q,r=aln.traceback.query,aln.traceback.ref
                if not q:continue
                identity=sum(a==b and a!='-' for a,b in zip(q,r))/len(q)
                if identity<.4:continue
                covq=sum(a!='-' for a in q)/lengths[i];covr=sum(a!='-' for a in r)/lengths[j]
                if covq>=.8 and covr>=.8:
                    out.append(dict(i=i,j=int(j),identity=float(identity),coverage_i=float(covq),coverage_j=float(covr),alignment_columns=len(q),score=int(aln.score)))
            return out,len(cand)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(scan,i) for i in range(len(sequences)-1)]
            for future in as_completed(futures):
                out,n=future.result();records.extend(out);scanned+=1;eligible+=n
                if scanned%200==0:
                    progress=dict(native_sequences=len(sequences),query_rows_complete=scanned,eligible_pairs_aligned=eligible,hits=len(records),elapsed_seconds=round(time.time()-started,1))
                    (ROOT/'similarity_progress.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
        edges=pd.DataFrame(records,columns=['i','j','identity','coverage_i','coverage_j','alignment_columns','score']).sort_values(['i','j']).reset_index(drop=True)
        edges.to_parquet(dest,index=False)
        (ROOT/'similarity_sequence_manifest.json').write_text(json.dumps(dict(sequences_sha256=hashlib.sha256('\n'.join(sequences).encode()).hexdigest(),
            aligned_pairs=eligible,native_sequences=len(sequences),parasail_version=parasail.__version__,
            algorithm='Smith-Waterman BLOSUM62, gap open 10, extend 1; exhaustive length/count-bound-eligible pairs',
            identity_rule='exact identity / traceback alignment columns',coverage_rule='non-gap residues / full length in EACH sequence',
            primary_identity=.6,sensitivity_identity=.4,minimum_bidirectional_coverage=.8),indent=2))
    groups={}
    for label,cut in [('primary',.6),('sensitivity',.4)]:
        u=Union(len(sequences));u.parent=base_parent.copy()
        for r in edges.loc[edges.identity.ge(cut)].itertuples():u.join(r.i,r.j)
        groups[label]=[f'{label}_g{u.root(i):05d}' for i in range(len(sequences))]
    table=pd.DataFrame(dict(sequence=sequences,length=lengths,sources=[';'.join(sorted(sources[s])) for s in sequences],
        genes=[';'.join(sorted(t[5:] for t in tokens[s] if t.startswith('gene:'))) for s in sequences],
        primary_group=groups['primary'],sensitivity_group=groups['sensitivity']))
    proportions=counts/lengths[:,None]
    table['entropy_bits']=-(np.where(proportions>0,proportions*np.log2(np.maximum(proportions,1e-300)),0)).sum(axis=1)
    table['low_complexity']=table.entropy_bits.lt(2.5)
    table.to_parquet(ROOT/'sequence_dependence_groups.parquet',index=False)
    info=dict(native_sequences=len(table),primary_groups=int(table.primary_group.nunique()),sensitivity_groups=int(table.sensitivity_group.nunique()),
        primary_similarity_edges=int(edges.identity.ge(.6).sum()),sensitivity_edges=len(edges),
        largest_primary_component=int(table.primary_group.value_counts().max()),largest_sensitivity_component=int(table.sensitivity_group.value_counts().max()),
        available_reference_snapshot='current counterparts, not proof of historic training/pretraining disjointness')
    for source in ['Guo','Yang','annotation_reference','ShortStop_positive']:
        sub=table.loc[table.sources.str.split(';').map(lambda x:source in x)]
        info[source]=dict(unique_sequences=len(sub),primary_groups=int(sub.primary_group.nunique()),low_complexity=int(sub.low_complexity.sum()))
    (ROOT/'SEQUENCE_DEPENDENCE_AUDIT.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
