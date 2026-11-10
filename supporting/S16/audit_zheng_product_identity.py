"""Audit exact candidate identity and source-table peptide sharing; no spectra replay."""
from pathlib import Path
from collections import Counter
import csv, hashlib, json
from openpyxl import load_workbook

P=Path(__file__).resolve().parent
D=P/'detection_evolution_prior/Zheng2023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    roster=P/'native_function_bridge/candidate_bridge.csv'
    table=D/'jci-133-159940-s011.xlsx';ms=D/'jci-133-159940-s013.xlsx'
    with roster.open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
    w=load_workbook(table,read_only=True,data_only=True);rows=list(w.active.values);w.close()
    records=[dict(zip(rows[0],r),excel_row=i) for i,r in enumerate(rows[1:],2)]
    for r in records:
        raw=r['AA-sequence'];assert raw.endswith('*') and '*' not in raw[:-1]
        r['sequence']=raw[:-1];assert len(r['sequence'])==r['AA-length']
    assert len(candidates)==60 and len(records)==29
    out=[]
    for c in candidates:
        hits=[r for r in records if r['sequence']==c['sequence']]
        samegene=[r for r in records if r['Gid'].split('.')[0]==c['gene_id'].split('.')[0]]
        out.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],sequence=c['sequence'],exact_source_records=len(hits),source_ORFs=';'.join(r['ORF-ID'].strip() for r in hits),same_gene_id_records=len(samegene),new_function_label_assigned=False))
    assert sum(r['exact_source_records'] for r in out)==0
    outfile=D/'CANDIDATE_PRODUCT_CONTEXT.csv'
    with outfile.open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,list(out[0]));wr.writeheader();wr.writerows(out)
    sheets=[];w=load_workbook(ms,read_only=True,data_only=True)
    for s in w:
        rows=list(s.values);headers=rows[0];idx=headers.index('Sequence')
        psms=[(i,r) for i,r in enumerate(rows[1:],2) if isinstance(r[idx],str) and r[idx]]
        counts=Counter(r[idx].upper() for i,r in psms)
        mapping=[]
        for seq,n in sorted(counts.items()):
            mapping.append(dict(sequence=seq,source_record_count=n,source_ORFs=[r['ORF-ID'].strip() for r in records if seq in r['sequence']],source_AA_lengths=[r['AA-length'] for r in records if seq in r['sequence']]))
        sheets.append(dict(sheet=s.title,source_record_count=len(psms),distinct_unmodified_sequences=len(counts),peptide_mapping=mapping))
    w.close()
    selected=['QERFPIILLSR','TDSFAGHLFSTAR','ATENTCTMVSGR','TLPLSVER']
    figure_peptides=[dict(sequence=seq,matches=[dict(ORF=r['ORF-ID'].strip(),length=r['AA-length'],position_1based=r['sequence'].index(seq)+1) for r in records if seq in r['sequence']]) for seq in selected]
    o=dict(sources={n:dict(path=str(p),sha256=sha(p)) for n,p in [('roster',roster),('Table1',table),('Table2',ms)]},selected_hit_ORFs=28,source_sequence_records=29,extra_record='RACE-supported 120-aa GENCODE v39 version of the original 63-aa ORF7;not an extra screen hit',candidates_checked=60,exact_candidate_matches=sum(r['exact_source_records'] for r in out),same_gene_id_matches=sum(r['same_gene_id_records'] for r in out),figure_peptide_source_mapping=figure_peptides,Table2=sheets,normalization='Remove only the single terminal * from Table1;uppercase Table2 residue letters to remove modification-case notation;no substitutions or initiator changes',scope='Sharing tested only against the 29 Table1 records;not whole-proteome uniqueness, spectral confidence, abundance or alternative-isoform expression',new_function_label_assigned=False,new_predictor_fit_or_inference=False,raw_spectra_reprocessed=False)
    (D/'PRODUCT_IDENTITY_AUDIT.json').write_text(json.dumps(o,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in o.items() if k not in ['sources','Table2','figure_peptide_source_mapping']},indent=2))
    print(json.dumps([dict(sheet=s['sheet'],records=s['source_record_count'],sequences=s['distinct_unmodified_sequences']) for s in sheets],indent=2))

if __name__=='__main__':main()
