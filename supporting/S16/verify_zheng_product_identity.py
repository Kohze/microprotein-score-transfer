"""Independent XLSX XML checks of the source identity audit."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json
from verify_cardiac_candidate_context import sheet_rows

P=Path(__file__).resolve().parent
D=P/'detection_evolution_prior/Zheng2023'
def main():
    a=json.loads((D/'PRODUCT_IDENTITY_AUDIT.json').read_text())
    for x in a['sources'].values():assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    source=sheet_rows(a['sources']['Table1']['path'],'sheet1')
    rs=[dict(row=i,orf=c[0].strip(),gene=c[1],seq=c[11][:-1],length=int(c[12])) for i,c in source if i>1 and c.get(11)]
    assert len(rs)==29 and all(len(r['seq'])==r['length'] for r in rs)
    assert sum(r['orf'].startswith('ORF-') for r in rs)==28
    with Path(a['sources']['roster']['path']).open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
    with (D/'CANDIDATE_PRODUCT_CONTEXT.csv').open(encoding='utf-8-sig',newline='') as f:out=list(csv.DictReader(f))
    assert len(candidates)==len(out)==60
    for c,o in zip(candidates,out):
        assert c['candidate_key']==o['candidate_key'] and c['sequence']==o['sequence']
        hits=[r for r in rs if r['seq']==c['sequence']]
        genes=[r for r in rs if r['gene'].split('.')[0]==c['gene_id'].split('.')[0]]
        assert len(hits)==int(o['exact_source_records'])==0
        assert len(genes)==int(o['same_gene_id_records'])==0
        assert o['new_function_label_assigned']=='False'
    for item in a['figure_peptide_source_mapping']:
        matches=[dict(ORF=r['orf'],length=r['length'],position_1based=r['seq'].index(item['sequence'])+1) for r in rs if item['sequence'] in r['seq']]
        assert matches==item['matches']
        assert 120 in [r['length'] for r in matches] and 131 in [r['length'] for r in matches]
    checked=0
    for s in a['Table2']:
        original=sheet_rows(a['sources']['Table2']['path'],s['sheet'])
        seqs=[c[3].upper() for i,c in original if i>1 and isinstance(c.get(3),str) and c[3]]
        counts=Counter(seqs);assert len(seqs)==s['source_record_count'] and len(counts)==s['distinct_unmodified_sequences']
        expected=[dict(sequence=q,source_record_count=n,source_ORFs=[r['orf'] for r in rs if q in r['seq']],source_AA_lengths=[r['length'] for r in rs if q in r['seq']]) for q,n in sorted(counts.items())]
        assert expected==s['peptide_mapping'];checked+=len(seqs)
    assert not a['new_function_label_assigned'] and not a['raw_spectra_reprocessed']
    result=dict(status='passed',independent_reader='Original XLSX ZIP/XML, no openpyxl',candidates_checked=60,source_sequence_records=29,source_screen_hits=28,exact_candidate_matches=0,same_gene_id_matches=0,Table2_source_records_checked=checked,figure_peptides_checked=4,sharing_scope='Table1 records only;not alternative-isoform expression or whole-proteome uniqueness',new_native_function_validation=False)
    (D/'PRODUCT_IDENTITY_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
