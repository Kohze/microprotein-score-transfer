"""Verify source joins and assay annotations with an independent XLSX XML reader."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
from verify_cardiac_candidate_context import sheet_rows

P=Path(__file__).resolve().parent;D=P/'detection_evolution_prior/Prensner2021'
def main():
    a=json.loads((D/'CANDIDATE_ASSAY_AUDIT.json').read_text())
    for x in a['sources'].values():assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    book=a['sources']['Tables1_17']['path'];raw=sheet_rows(book,'Table S2')
    rows=[dict(row=i,name=c[0],seq=c[52].removesuffix('*'),rawseq=c[52],length=int(c[53]),v5=c.get(40),crispr=c.get(41),both=c.get(44),RR=c.get(42),TAS=c.get(43),mean=c.get(39),gene=c.get(70)) for i,c in raw if i>=6 and c.get(0)]
    assert len(rows)==553 and all(len(x['rawseq'])==x['length'] for x in rows)
    lengths=[len(x['seq']) for x in rows]
    assert dict(min=min(lengths),max=max(lengths),residues31_150=sum(31<=n<=150 for n in lengths),below31=sum(n<31 for n in lengths),above150=sum(n>150 for n in lengths))==a['source_residue_length_scope']
    assert sum(x['v5']==1 for x in rows)==a['source_summary_flag_counts']['V5 validation']==257
    assert sum(x['crispr']==1 for x in rows)==a['source_summary_flag_counts']['CRISPR phenotype']==57
    assert sum(x['both']==1 for x in rows)==a['source_summary_flag_counts']['Both RR and TAS']==237
    stops=[dict(name=x['name'],raw_string_length=len(x['rawseq']),residue_length=len(x['seq']),source_length=x['length']) for x in rows if x['rawseq'].endswith('*')];assert stops==a['terminal_stop_notation_records'] and len(stops)==3
    with Path(a['sources']['roster']['path']).open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
    with (D/'CANDIDATE_ASSAY_CONTEXT.csv').open(encoding='utf-8-sig',newline='') as f:output=list(csv.DictReader(f))
    assert len(candidates)==len(output)==60;found=[];gene_only=[]
    for c,o in zip(candidates,output):
        assert c['candidate_key']==o['candidate_key'] and c['sequence']==o['sequence']
        exact=[x for x in rows if x['seq']==c['sequence']];same=[x for x in rows if isinstance(x['gene'],str) and x['gene'].split('.')[0]==c['gene_id'].split('.')[0] and x not in exact]
        assert len(exact)==int(o['exact_source_matches']) and len(same)==int(o['gene_ID_only_different_sequence_records'])
        assert sum(x['v5']==1 for x in exact)==int(o['source_V5_flagged'])
        assert sum(x['crispr']==1 for x in exact)==int(o['source_CRISPR_flagged'])
        assert sum(x['both']==1 for x in exact)==int(o['source_RR_and_TAS_flagged'])
        assert int(c['exact_primary_GFP_records'])==int(o['primary_GFP_records']) and int(c['exact_measured_edit_pairs'])==int(o['existing_measured_edit_pairs'])
        assert o['new_native_function_label_assigned']=='False'
        found += [(c,x) for x in exact];gene_only += [(c['candidate_key'],x['name'],x['row']) for x in same]
    assert len(found)==11 and sum(x['v5']==1 for c,x in found)==a['matched_V5_flagged']==5
    assert sum(x['both']==1 for c,x in found)==a['matched_RR_and_TAS_flagged']==8
    assert sum(x['crispr']==1 for c,x in found)==a['matched_CRISPR_flagged']==0
    assert sum(int(c['exact_primary_GFP_records'])>0 for c,x in found)==a['matches_with_primary_reporter']==6
    assert sorted(gene_only)==sorted((x['candidate_key'],x['source_name'],x['source_excel_row']) for x in a['gene_ID_only_different_product_records'])
    s16=sheet_rows(book,'Table S16');checked=0
    for c,x in found:
        d=next(d for d in a['exact_match_details'] if d['candidate_key']==c['candidate_key'] and d['source_excel_row']==x['row'])
        assert x['rawseq']==d['source_raw_sequence'] and x['RR']==d['source_max_RR'] and x['TAS']==d['source_max_TAS']
        label=a['source_TableS16_label_aliases'].get(x['name'],x['name'])
        selected=[dict(excel_row=i,kind=v.get(0),label=v.get(1),replicates=[v.get(k) for k in [3,4,5]],mean=v.get(6),above_noise=v.get(7)) for i,v in s16 if v.get(1)==label and v.get(0) in ['Wild_Type','Paired_Mutant']]
        assert selected==d['TableS16_rows'];checked+=len(selected)
    assert checked==a['source_TableS16_rows_checked']==14
    rawtiling=sheet_rows(a['sources']['Tables21_32']['path'],'Table S32');classes=Counter(v[1] for i,v in rawtiling if i>=5 and v.get(0));assert sum(classes.values())==41 and dict(classes)==a['source_tiling_class_counts']
    assert not a['new_predictor_fit_or_inference'] and not a['new_native_function_label_assigned'] and not a['new_native_or_edited_function_measurement']
    out=dict(status='passed',independent_reader='Original XLSX ZIP/XML and CSV;no openpyxl',source_ORFs_checked=553,candidates_checked=60,exact_candidate_records=11,matched_V5_flagged=5,matched_RR_and_TAS_flagged=8,matched_CRISPR_flagged=0,matches_with_primary_reporter=6,TableS16_source_rows_checked=14,gene_ID_only_different_product_records=len(gene_only),terminal_stop_notation_records=3,source_tiling_records_checked=41,new_native_or_edited_function_measurement=False)
    (D/'CANDIDATE_ASSAY_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
