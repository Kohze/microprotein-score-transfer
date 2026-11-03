"""Exclude source-linked independent C-terminal pairs before outcome association."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
B=P.parent/'stability_benchmark_2026-10-31'
O=P/'independent_interventions'
sys.path.insert(0,str(B/'vendor'))
import parasail


def main():
    d=pd.read_parquet(O/'sequence_pair_audit.parquet')
    all_source=pd.read_parquet(B/'sequence_dependence_groups.parquet')
    source=all_source[all_source.sources.str.contains('Guo|annotation_reference',regex=True)]
    seqs=source.sequence.tolist()
    lengths=np.array(list(map(len,seqs)))
    aa='ACDEFGHIKLMNPQRSTVWY'
    counts=np.array([[s.count(a) for a in aa] for s in seqs],dtype=np.int16)
    targets=sorted(set(d.wt_sequence)|set(d.variant_sequence))
    began=time.time()
    def scan(seq):
        count=np.array([seq.count(a) for a in aa])
        ix=np.flatnonzero((lengths>=.8*len(seq))&(.8*lengths<=len(seq)))
        bound=np.minimum(count,counts[ix]).sum(1)
        ix=ix[bound+1e-9>=.32*np.maximum(lengths[ix],len(seq))]
        hits=[]
        for j in ix:
            result=parasail.sw_trace_striped_16(seq,seqs[j],10,1,parasail.blosum62)
            q,r=result.traceback.query,result.traceback.ref
            if not q:
                continue
            ident=sum(a==b and a!='-' for a,b in zip(q,r))/len(q)
            if ident<.4:
                continue
            cq,cr=sum(a!='-' for a in q)/len(seq),sum(a!='-' for a in r)/len(seqs[j])
            if cq>=.8 and cr>=.8:
                hits.append(dict(target_sequence=seq,source_sequence=seqs[j],identity=ident,
                    coverage_target=cq,coverage_source=cr))
        return hits,len(ix)
    edges=[]
    aligned=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(scan,seq) for seq in targets]
        for i,job in enumerate(as_completed(jobs),start=1):
            hits,n=job.result()
            edges.extend(hits)
            aligned+=n
            if i%500==0:
                progress=dict(completed=i,total=len(targets),aligned_pairs=aligned,edges=len(edges),elapsed_seconds=round(time.time()-began,1))
                (O/'SEPARATION_PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n')
                print(json.dumps(progress),flush=True)
    edge=pd.DataFrame(edges,columns=['target_sequence','source_sequence','identity','coverage_target','coverage_source'])
    edge.to_parquet(O/'source_similarity_edges.parquet',index=False)
    valid=(d.standard_sequences&d.wt_length.eq(58)&d.variant_length.eq(58)&d.wt_endswith_shared&
           ~d.identical_pair&d.unique_gene_mapping&d.reported_RT_length_consistent&~d.source_gene_overlap)
    pap=(~d.PAP_training_gene_overlap&~d.PAP_training_exact30_window&~d.human25_training_exact25_window)
    for label,cut in [('primary60',.6),('sensitivity40',.4)]:
        blocked=set(edge.loc[edge.identity.ge(cut),'target_sequence'])
        d[label+'_source_similar']=d.wt_sequence.isin(blocked)|d.variant_sequence.isin(blocked)
        d[label+'_head_only_eligible']=valid&~d[label+'_source_similar']
        d[label+'_common_eligible']=valid&pap&~d[label+'_source_similar']
    # Cluster repeated genes and identical measured cargos across target pairs.
    parent=list(range(len(d)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    seen={}
    for i,row in enumerate(d.itertuples()):
        keys=['gene:'+row.mapped_gene_ids] if row.unique_gene_mapping else []
        keys+=['seq:'+row.wt_sequence,'seq:'+row.variant_sequence]
        for key in keys:
            if key in seen:
                union(i,seen[key])
            else:
                seen[key]=i
    d['target_dependence_group']=[f'CHU{root(i):05d}' for i in range(len(d))]
    d.to_parquet(O/'separated_sequence_pairs.parquet',index=False)
    receipt=dict(status='outcome-free operational source separation complete; measurement audit and prediction freeze remain',
        protocol_sha256=hashlib.sha256((O/'COMPARISON_PROTOCOL.txt').read_bytes()).hexdigest(),
        source_manifest_sha256=hashlib.sha256((B/'sequence_dependence_groups.parquet').read_bytes()).hexdigest(),
        targets=len(targets),aligned_pairs=aligned,edges=len(edge),
        eligible_counts={col:int(d[col].sum()) for col in d if col.endswith('_eligible')},
        primary_common_groups=int(d.loc[d.primary60_common_eligible,'target_dependence_group'].nunique()),
        target_outcomes_loaded=False,uncertainty_groups='gene plus identical cargos; not exhaustive homology families')
    (O/'SOURCE_SEPARATION_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    main()
