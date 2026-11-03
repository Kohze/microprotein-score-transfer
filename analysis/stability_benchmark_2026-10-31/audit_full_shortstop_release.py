"""Complete release translation, independent of Guo identifiers and outcomes."""
from pathlib import Path
import sys,re,json,hashlib,shutil
from collections import defaultdict
import pandas as pd
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'full_database_study_2026-09-13/vendor'))
from twobitreader import TwoBitFile
CODONS=dict(zip((a+b+c for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
def main():
    gtf=P/'inputs/ShortStop2025_code/gencode_smorfs.gtf'
    if not gtf.exists():shutil.copy2(P.parent/'deep_novelty_2026-10-31/ShortStop2025_code/gencode_smorfs.gtf',gtf)
    pred=pd.read_csv(P/'inputs/sources/shortstop_v1.2_gencode_smorfs_classifications.csv');byid=defaultdict(list)
    for line in gtf.read_text().splitlines():
        q=line.split('\t')
        if len(q)==9 and q[2]=='CDS':byid[re.search(r'gene_id ([^;]+);',q[8]).group(1)].append((q[0],int(q[3])-1,int(q[4]),q[6]))
    genome=TwoBitFile(str(P.parent/'full_database_study_2026-09-13/references/hg38.2bit'));rows=[]
    for ident in pred.orf_id:
        b=byid[ident];assert b and len({(x[0],x[3]) for x in b})==1;chrom,_,_,strand=b[0];dna=''.join(genome[chrom][a:z].upper() for _,a,z,_ in sorted(b,key=lambda x:x[1]))
        if strand=='-':dna=dna.translate(str.maketrans('ACGTN','TGCAN'))[::-1]
        seq=''.join(CODONS.get(dna[j:j+3],'X') for j in range(0,len(dna)-2,3)).rstrip('*');rows.append(dict(orf_id=ident,sequence=seq,valid=len(dna)%3==0 and '*' not in seq and 'X' not in seq))
    d=pd.DataFrame(rows).merge(pred,on='orf_id',validate='one_to_one');d.to_parquet(P/'inputs/ShortStop_full_release_sequence_verified.parquet',index=False)
    y=pd.read_parquet(P/'validation_assay_records.parquet');z=d[d.valid];summary=[]
    for row in y.itertuples():
        matches=z[z.sequence.eq(row.native_sequence)];summary.append(dict(orf_name=row.orf_name,n_matched_release_ids=len(matches),n_distinct_scores=int(matches.mean_sam_probability.nunique()),orf_ids=';'.join(matches.orf_id)))
    pd.DataFrame(summary).to_csv(P/'results/Yang_full_ShortStop_coverage_audit.csv',index=False)
    receipt=dict(release_rows=len(d),valid_translations=int(d.valid.sum()),Yang_peptides_with_match=sum(r['n_matched_release_ids']>0 for r in summary),Yang_unambiguous_score_matches=sum(r['n_distinct_scores']==1 for r in summary),gtf_sha256=hashlib.sha256(gtf.read_bytes()).hexdigest(),prepared_table_sha256=hashlib.sha256((P/'inputs/ShortStop_full_release_sequence_verified.parquet').read_bytes()).hexdigest(),scope='Late coverage audit of entire frozen release, no labels used in translation/mapping; no predictor fitting or validation selection.')
    (P/'results/FULL_SHORTSTOP_RELEASE_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
