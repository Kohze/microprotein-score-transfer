"""Verify native joins from workbooks and reference mapping from FASTA."""
from pathlib import Path
import json
import math
import re
from openpyxl import load_workbook

P=Path(__file__).resolve().parent
D=P/'native_system_prior/candidate_primaries'
B=P/'native_function_bridge'


def seq(x):
    if not isinstance(x,str):return ''
    return ''.join(x.split()).removesuffix('*')


def rows(book,sheet,header):
    allrows=list(book[sheet].values)
    return [(i+1,dict(zip(allrows[header-1],r)),r) for i,r in enumerate(allrows) if i>=header and isinstance(r[14] if len(r)>14 else None,str)]


def same(a,b):
    if a is None:return b==''
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        return math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-12)
    return a==b


def main():
    original=load_workbook(B/'TableS2_publisher.xlsx',read_only=True,data_only=True)
    cs=list(original['PRISMA candidates summary'].values)
    cands=[dict(zip(cs[0],r)) for r in cs[1:] if any(x is not None for x in r)]
    out=json.loads((D/'candidate_native_evidence.json').read_text())
    assert len(cands)==len(out)==60
    w5=load_workbook(D/'HumanHeart2019_mmc5_candidate.xlsx',read_only=True,data_only=True)
    w3=load_workbook(D/'HumanHeart2019_mmc3_candidate.xlsx',read_only=True,data_only=True)
    source=[('S5',i,d) for i,d,_ in rows(w5,'HS left ventricle (hg38)',2)]
    source += [('S3',i,d) for i,d,_ in rows(w3,'Human heart uORFs',2)]
    set_rows=list(w5['SRM set up'].values)[1:]
    result_rows=list(w5['SRM Results'].values)
    primer_rows=list(w5['IVT assay mutagenesis primers'].values)
    checked_fields=0;checked_runs=0;links=0
    for raw,c in zip(cands,out):
        expected_seq=seq(raw['Peptide Sequence'])
        assert expected_seq==c['sequence'] and raw['Candidate name']==c['candidate_name']
        matches=[(table,i,d) for table,i,d in source if seq(d['ORF peptide sequence (aa)' if table=='S5' else 'uORF peptide sequence (aa)'])==expected_seq]
        assert [(t,i) for t,i,_ in matches]==[(x['table'],x['excel_row']) for x in c['source_matches']]
        for (table,i,d),result in zip(matches,c['source_matches']):
            links+=1
            for k,v in d.items():
                if k is None:continue
                assert same(v,result['raw_source_fields'][str(k).strip()]),(c['candidate_name'],k,v)
                checked_fields+=1
            if table!='S5':continue
            oid=d['ORF ID (transcriptID_ORFstart_ORFstop)']
            target=[r for r in set_rows if oid in str(r[1]).split(';') and seq(r[2]) in expected_seq]
            assert len(target)==result['SRM_target_transition_rows']
            assert sorted({r[2] for r in target})==result['SRM_target_peptides']
            expected=[]
            rr=[]
            for excelrow,r in enumerate(result_rows,1):
                if len(r)<43 or r[1]!=oid or seq(r[2])!=expected_seq:continue
                rr.append(excelrow)
                for j in range(10):
                    v=r[3+4*j:7+4*j]
                    if isinstance(v[0],str) and v[0].upper() not in ['FALSE','NA','']:
                        expected.append((excelrow,j//2+1,j%2+1,*v))
            actual=[(r['excel_row'],r['heart'],r['technical_replicate'],r['peptide'],r['dot_product'],r['total_area'],r['retention_time']) for r in result['SRM_reported_positive_runs']]
            assert expected==actual,(c['candidate_name'],expected,actual)
            assert len(rr)==result['SRM_full_sequence_result_rows']
            checked_runs+=len(expected)
            prim=[j for j,r in enumerate(primer_rows,1) if len(r)>3 and r[2]==oid and seq(r[3])==expected_seq]
            assert prim==result['IVT_mutagenesis_source_rows']
            status='reported detected' if expected else 'targeted, not reported detected' if target and rr else 'unavailable or not targeted'
            assert status==result['SRM_status']
    # Independent representation and matching algorithm: FASTA plus codon regex.
    fa=(D/'NR_024451_1.fasta').read_text().splitlines()
    assert fa[0].startswith('>NR_024451.1')
    nt=''.join(fa[1:]).upper()
    codons={
        'A':'GCT GCC GCA GCG','C':'TGT TGC','D':'GAT GAC','E':'GAA GAG',
        'F':'TTT TTC','G':'GGT GGC GGA GGG','H':'CAT CAC','I':'ATT ATC ATA',
        'K':'AAA AAG','L':'TTA TTG CTT CTC CTA CTG','M':'ATG','N':'AAT AAC',
        'P':'CCT CCC CCA CCG','Q':'CAA CAG','R':'CGT CGC CGA CGG AGA AGG',
        'S':'TCT TCC TCA TCG AGT AGC','T':'ACT ACC ACA ACG','V':'GTT GTC GTA GTG',
        'W':'TGG','Y':'TAT TAC'}
    peptide=next(x['sequence'] for x in out if x['candidate_name']=='JHDM1D-AS1')
    pattern=''.join('(?:'+'|'.join(codons[aa].split())+')' for aa in peptide)+'(?:TAA|TAG|TGA)'
    matches=[m for m in re.finditer(pattern,nt)]
    ref=json.loads((D/'JHDM_REFERENCE_COMPATIBILITY.json').read_text())
    assert len(matches)==len(ref['hits'])==1
    for match,hit in zip(matches,ref['hits']):
        assert match.start()+1==hit['start_1based']
        assert match.end()-3==hit['last_sense_nt_1based']
        assert match.end()==hit['stop_end_1based']
    assert len(nt)==ref['reference_length_nt']==2398
    assert not ref['historical_plasmid_sequence_or_translation_verified']
    assert not ref['reference_vs_heart_isoform_identity_established']
    # Literal peptide/edited sequence coordinates and overlap endpoints.
    import csv
    with (B/'motif_associated_tile_edit_context.csv').open(encoding='utf-8',newline='') as h: tiles=list(csv.DictReader(h))
    for region in ref['SRM_signature_edit_and_tile_context']:
        start,end=region['start_1based']-1,region['end_1based']
        assert peptide[start:end]==region['SRM_target_peptide']
        edited=peptide.replace('L','A') if region['edit_family']=='L_to_A' else peptide+'DNYNEPKANQ'
        assert edited[start:end]==region['edited_region']
        overlaps=[]
        for t in tiles:
            if t['candidate_name']!='JHDM1D-AS1' or t['mutation_group']!=region['edit_family']:continue
            low=max(start+1,int(t['tile_start_1based']));high=min(end,int(t['tile_end_1based']))
            if low<=high:overlaps.append((t['PRISMA_tile_ID'],low,high))
        assert overlaps==[(x['tile_ID'],x['overlap_start_1based'],x['overlap_end_1based']) for x in region['binding_tile_overlaps']]
    receipt=dict(status='passed',source_candidates_checked=60,exact_source_links_checked=links,
        original_workbook_field_values_checked=checked_fields,original_SRM_positive_run_records_checked=checked_runs,
        readers='Original Sandmann S2 and human-heart S3/S5 openpyxl vs pandas analysis',
        reference_verifier='Independent NCBI FASTA and amino-acid codon regex vs GenBank three-frame translation',
        exact_reference_ORFs_checked=1,signature_edit_region_records_checked=4,
        source_spectral_raw_data_or_function_verified=False,plasmid_sequence_verified=False)
    (D/'INDEPENDENT_NATIVE_EVIDENCE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
