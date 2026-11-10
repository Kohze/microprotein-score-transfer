"""Exact source ORF joins and assay annotations; no model or native-function inference."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
from openpyxl import load_workbook

P=Path(__file__).resolve().parent;D=P/'detection_evolution_prior/Prensner2021'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_rows(path,sheet,header):
    w=load_workbook(path,read_only=True,data_only=True);rows=list(w[sheet].values);w.close()
    return [dict(zip(rows[header-1],r),excel_row=i) for i,r in enumerate(rows[header:],header+1) if r[0] is not None]

def main():
    roster=P/'native_function_bridge/candidate_bridge.csv';book=D/'41587_2020_806_MOESM3_ESM.xlsx';tiling=D/'41587_2020_806_MOESM5_ESM.xlsx'
    with roster.open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
    records=source_rows(book,'Table S2',5);assert len(records)==553
    for r in records:
        raw=r['Amino Acid'];assert len(raw)==r['Amino acid length']
        r['sequence']=raw.removesuffix('*');assert '*' not in r['sequence']
    aliases={'MKKS uORF':'MKKS','MIEF1 uORF':'MIEF1'}
    w=load_workbook(book,read_only=True,data_only=True);s16=list(w['Table S16'].values);w.close()
    result=[];details=[];gene_only=[]
    for c in candidates:
        hits=[r for r in records if r['sequence']==c['sequence']]
        genes=[r for r in records if isinstance(r['Ensembl/GENCODE Gene ID (v19)'],str) and r['Ensembl/GENCODE Gene ID (v19)'].split('.')[0]==c['gene_id'].split('.')[0]]
        different=[r for r in genes if r not in hits]
        result.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],sequence=c['sequence'],length=len(c['sequence']),exact_source_matches=len(hits),source_names=';'.join(r['Name'] for r in hits),source_V5_flagged=sum(r['V5 validation']==1 for r in hits),source_CRISPR_flagged=sum(r['CRISPR phenotype']==1 for r in hits),source_RR_and_TAS_flagged=sum(r['Both RR and TAS']==1 for r in hits),gene_ID_only_different_sequence_records=len(different),primary_GFP_records=int(c['exact_primary_GFP_records']),existing_measured_edit_pairs=int(c['exact_measured_edit_pairs']),new_native_function_label_assigned=False))
        for r in hits:
            label=aliases.get(r['Name'],r['Name'])
            rows=[dict(excel_row=i,kind=x[0],label=x[1],replicates=list(x[3:6]),mean=x[6],above_noise=x[7]) for i,x in enumerate(s16,1) if x[1]==label and x[0] in ['Wild_Type','Paired_Mutant']]
            wild=[x for x in rows if x['kind']=='Wild_Type'];assert len(wild)==1
            assert (wild[0]['above_noise']=='Yes')==(r['V5 validation']==1)
            assert math.isclose(wild[0]['mean'],r['Average in cell western signal'],abs_tol=1e-10)
            details.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],length=len(c['sequence']),candidate_gene_id=c['gene_id'],source_gene_id=r['Ensembl/GENCODE Gene ID (v19)'],source_name=r['Name'],source_excel_row=r['excel_row'],source_raw_sequence=r['Amino Acid'],source_V5_flag=r['V5 validation'],source_CRISPR_flag=r['CRISPR phenotype'],source_RR_and_TAS_flag=r['Both RR and TAS'],source_max_RR=r['Maximum replicate correlation (expression profiling)'],source_max_TAS=r['Maximum transcriptional activity score'],primary_GFP_records=int(c['exact_primary_GFP_records']),existing_measured_edit_pairs=int(c['exact_measured_edit_pairs']),TableS16_label=label,TableS16_rows=rows))
        for r in different:gene_only.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],candidate_sequence=c['sequence'],candidate_length=len(c['sequence']),source_name=r['Name'],source_sequence=r['sequence'],source_length=len(r['sequence']),source_excel_row=r['excel_row'],function_transferred=False))
    assert len(candidates)==60 and len(details)==11
    with (D/'CANDIDATE_ASSAY_CONTEXT.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,list(result[0]));writer.writeheader();writer.writerows(result)
    tiling_rows=source_rows(tiling,'Table S32',4);assert len(tiling_rows)==41
    flags={key:sum(r[key]==1 for r in records) for key in ['V5 validation','CRISPR phenotype','Both RR and TAS']}
    stop=[dict(name=r['Name'],raw_string_length=len(r['Amino Acid']),residue_length=len(r['sequence']),source_length=r['Amino acid length']) for r in records if r['Amino Acid'].endswith('*')]
    lengths=[len(r['sequence']) for r in records]
    length_scope=dict(min=min(lengths),max=max(lengths),residues31_150=sum(31<=n<=150 for n in lengths),below31=sum(n<31 for n in lengths),above150=sum(n>150 for n in lengths))
    out=dict(sources={n:dict(path=str(p),sha256=sha(p)) for n,p in [('roster',roster),('Tables1_17',book),('Tables21_32',tiling)]},source_ORFs=553,source_residue_length_scope=length_scope,source_summary_flag_counts=flags,terminal_stop_notation_records=stop,candidates_checked=60,exact_candidate_records=11,matched_V5_flagged=sum(x['source_V5_flag']==1 for x in details),matched_RR_and_TAS_flagged=sum(x['source_RR_and_TAS_flag']==1 for x in details),matched_CRISPR_flagged=sum(x['source_CRISPR_flag']==1 for x in details),matches_with_primary_reporter=sum(x['primary_GFP_records']>0 for x in details),matches_with_existing_edits=sum(x['existing_measured_edit_pairs']>0 for x in details),source_TableS16_rows_checked=sum(len(x['TableS16_rows']) for x in details),source_TableS16_label_aliases=aliases,exact_match_details=details,gene_ID_only_different_product_records=gene_only,source_tiling_class_counts=dict(Counter(r['Phenotype type'] for r in tiling_rows)),scope='Exact complete source peptide sequence after removal only of terminal stop notation;no initiator adjustment,residue substitution or substring acceptance. TableS16 cross-references use two explicit source label aliases. Source assay flags are annotations,not new endogenous function or measurements of our stability edits.',new_predictor_fit_or_inference=False,new_native_function_label_assigned=False,new_native_or_edited_function_measurement=False)
    (D/'CANDIDATE_ASSAY_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['sources','exact_match_details','gene_ID_only_different_product_records']},indent=2))

if __name__=='__main__':main()
