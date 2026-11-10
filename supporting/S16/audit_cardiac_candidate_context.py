"""Join all predefined candidates to the original young-sORF table by peptide."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,openpyxl

P=Path(__file__).resolve().parent
D=P/'detection_evolution_prior'
S=D/'CardiacEvolution2024'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    candidate_path=P/'native_function_bridge/candidate_bridge.csv'
    with candidate_path.open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
    assert len(candidates)==60 and len({r['candidate_key'] for r in candidates})==60
    table_path=S/'44161_2024_544_MOESM3_ESM.xlsx'
    w=openpyxl.load_workbook(table_path,read_only=True,data_only=True)
    rr=list(w['Supplementary Table 6'].iter_rows(values_only=True));headers=rr[1][:8]
    records=[dict(zip(headers,row[:8]),source_excel_row=n) for n,row in enumerate(rr[2:],3) if row[0] is not None]
    w.close();assert len(records)==504
    assert all(len(r['sequence'])==int(r['length']) for r in records)
    out=[];details=[]
    for c in candidates:
        # Same full amino-acid sequence; remove only source terminal-stop notation.
        # Do not change an initiator, substitute residues, or accept partial matches.
        hits=[r for r in records if c['sequence']==r['sequence'].rstrip('*')]
        human=[r for r in hits if r['species']=='human'];other=[r for r in hits if r['species']!='human']
        out.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],sequence=c['sequence'],length=c['length'],source_candidate_gene_id=c['gene_id'],exact_young_table_human_matches=len(human),human_cardiac_ORFs=';'.join(r['orf'] for r in human),human_age_labels=';'.join(r['age'] for r in human),human_cardiac_gene_ids=';'.join(r['gene_id'] for r in human),nonhuman_sequence_matches=len(other),nonhuman_match_species=';'.join(r['species'] for r in other),exact_primary_GFP_records=c['exact_primary_GFP_records'],exact_measured_edit_pairs=c['exact_measured_edit_pairs'],native_function_label_assigned=False))
        for r in hits:details.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],source_candidate_gene_id=c['gene_id'],sequence=c['sequence'],candidate_exact_primary_GFP_records=int(c['exact_primary_GFP_records']),candidate_exact_measured_edit_pairs=int(c['exact_measured_edit_pairs']),cardiac=r,human_evidence=r['species']=='human'))
    with (D/'CARDIAC_CANDIDATE_CONTEXT.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,out[0].keys());writer.writeheader();writer.writerows(out)
    source_path=S/'44161_2024_544_MOESM8_ESM.xlsx'
    w=openpyxl.load_workbook(source_path,read_only=True,data_only=True);knockdowns={}
    for target in ['LINC01405','SRP14-AS1']:
        rr=list(w['CRISPRi '+target].iter_rows(values_only=True))
        values=[dict(gene=r[0],padj=r[1],log2FC=r[2]) for r in rr[1:] if isinstance(r[0],str) and isinstance(r[1],(int,float)) and isinstance(r[2],(int,float))]
        significant=[r for r in values if r['padj']<.05 and abs(r['log2FC'])>.5]
        downstream=[r for r in significant if r['gene']!=target]
        nontext=[dict(excel_row=n,cached_gene_value=str(r[0]),gene_value_type=type(r[0]).__name__,padj=r[1],log2FC=r[2]) for n,r in enumerate(rr[1:],2) if r[0] is not None and not isinstance(r[0],str)]
        knockdowns[target]=dict(numeric_rows=len(values),numeric_row_scope='Numeric p/fold-change values with text gene names;non-text cached gene cells listed separately',nontext_gene_rows=nontext,threshold='padj<0.05 and abs(log2FC)>0.5;target gene excluded from downstream count',significant_including_target=len(significant),downstream_up=sum(r['log2FC']>.5 for r in downstream),downstream_down=sum(r['log2FC']<-.5 for r in downstream),selected_genes=[r for r in values if r['gene'] in ['LINC01405','SRP14-AS1','MYL2','FBN2']],significant_downstream_genes=downstream)
    w.close()
    result=dict(scope='Exact peptide links to selected young-sORF table and source DE-table count check;not new native functional or independent reporter validation',sources=dict(candidate_roster=dict(path=str(candidate_path),sha256=sha(candidate_path)),young_table=dict(path=str(table_path),sha256=sha(table_path)),source_data_Fig5=dict(path=str(source_path),sha256=sha(source_path))),young_table_records=len(records),young_table_species_counts=dict(Counter(r['species'] for r in records)),young_table_age_counts=dict(Counter(r['age'] for r in records)),candidate_roster_records=len(candidates),candidate_records_with_human_exact_match=sum(r['exact_young_table_human_matches']>0 for r in out),human_exact_match_records=sum(r['exact_young_table_human_matches'] for r in out),nonhuman_sequence_match_records=sum(r['nonhuman_sequence_matches'] for r in out),human_matching_candidate_gene_ids=len({r['source_candidate_gene_id'] for r in out if r['exact_young_table_human_matches']}),human_matches_with_primary_reporter_records=sum(r['exact_young_table_human_matches']>0 and int(r['exact_primary_GFP_records'])>0 for r in out),matched_details=details,CRISPRi_source_table_audit=knockdowns,negative_match_interpretation='No exact match in the selected young table;not evidence of no cardiac translation,absence from preserved ORFs or no biological function',cross_species_interpretation='An exact chimp peptide match is not human cardiac evidence or transfer of chimp locus evolutionary age',human_LV_source_reused_from_vanHeesch2019=True,new_predictor_fit_or_inference=False,new_native_function_label=False)
    (D/'CARDIAC_CANDIDATE_AND_PERTURBATION_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['matched_details','CRISPRi_source_table_audit']},indent=2))

if __name__=='__main__':main()
