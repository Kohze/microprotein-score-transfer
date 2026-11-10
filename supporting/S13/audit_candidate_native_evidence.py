"""Exact native-source joins and a reference-transcript compatibility check."""
from pathlib import Path
from itertools import product
import hashlib
import json
import re
import pandas as pd

P=Path(__file__).resolve().parent
D=P/'native_system_prior/candidate_primaries'
BRIDGE=P/'native_function_bridge'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(x):
    if not isinstance(x,str):return ''
    x=''.join(x.split())
    return x[:-1] if x.endswith('*') else x


def frame(sheet,header):
    d=pd.read_excel(sheet[0],sheet_name=sheet[1],header=header,keep_default_na=False)
    d.columns=[str(x).strip() for x in d.columns]
    return d


def safe(v):
    if hasattr(v,'item'):return v.item()
    return v


def main():
    (D/'NATIVE_EVIDENCE_JOIN_PROTOCOL.txt').write_text('''EXPLORATORY NATIVE EVIDENCE / REFERENCE COMPATIBILITY AUDIT
All60 Sandmann full-sequence candidates are retained. The three motif-linked
candidates and selected source rows have already been inspected; this is
post-hoc descriptive provenance, not a blinded validation or enrichment test.
Join HumanHeart2019 S3/S5 only by full amino-acid identity (whitespace and one
terminal stop normalized). Gene/ORF names annotate and check a sequence join;
they cannot substitute for it. Preserve source rows, original values and
all unmatched candidates. Source ORF lengths are nucleotides, not residues.

For S5 SRM, use exact full source sequence and source ORF ID for results.
Set-up IDs may list multiple ORFs separated by semicolons; explicitly split
those fields and require a targeted peptide to occur in the matched WT.
Separate reported detection, targeted nondetection and unavailable/untested
coverage. Retain every dot product/area/retention-time cell. The source's
dot-product threshold is0.7; no threshold is selected here. No raw spectral
reanalysis, calibrated FDR, native half-life or functional label is supplied.

Map the exact68aa JHDM1D-AS1 peptide into all three plus-strand frames of
NR_024451.1, cited by the2017 RNA study. Require ATG plus immediate in-frame
stop. Report1-based coordinates from the reference, not equivalence of the
951nt heart assembly and2398nt RefSeq transcript or of historical plasmids.
Map reported SRM peptide and source binding tiles within the exact WT and
compare their literal sequences under each existing edit. Sequence overlap
is not kinase causality, edited phosphorylation or interaction loss.
''',encoding='utf-8')
    s5=D/'HumanHeart2019_mmc5_candidate.xlsx'
    s3=D/'HumanHeart2019_mmc3_candidate.xlsx'
    hs=frame((s5,'HS left ventricle (hg38)'),1)
    us=frame((s3,'Human heart uORFs'),1)
    setup=frame((s5,'SRM set up'),0)
    results=frame((s5,'SRM Results'),2)
    primers=frame((s5,'IVT assay mutagenesis primers'),0)
    candidates=pd.read_csv(BRIDGE/'candidate_bridge.csv',keep_default_na=False)
    records=[]
    for n,df,pepcol,idcol in [
        ('S5',hs,'ORF peptide sequence (aa)','ORF ID (transcriptID_ORFstart_ORFstop)'),
        ('S3',us,'uORF peptide sequence (aa)','uORF ID (transcriptID_ORFstart_ORFstop)')]:
        for i,row in df.iterrows():
            seq=clean(row[pepcol])
            if not seq or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]+',seq):continue
            records.append(dict(table=n,excel_row=int(i)+3,sequence=seq,orf_id=row[idcol],
                raw_source_fields={str(k):safe(v) for k,v in row.items()}))
    roster=[]
    for _,c in candidates.iterrows():
        matches=[]
        for row in records:
            if clean(c['sequence'])!=row['sequence']:continue
            r=dict(row)
            if row['table']=='S5':
                targeted=setup.loc[setup['ORF ID'].map(lambda v: row['orf_id'] in str(v).split(';'))]
                peptide_rows=targeted.loc[targeted['Peptide sequence'].map(lambda v: clean(v) in row['sequence'])]
                rr=results.loc[(results['ORF ID (transcriptID_ORFstart_ORFstop)']==row['orf_id']) &
                    results['ORF peptide sequence (aa)'].map(lambda v:clean(v)==row['sequence'])]
                run_records=[]
                for idx,sr in rr.iterrows():
                    for run in range(10):
                        peptide,dot,area,retention=sr.iloc[3+run*4:7+run*4]
                        if isinstance(peptide,str) and peptide and peptide.upper() not in ['FALSE','NA']:
                            peptide_list=[clean(x) for x in peptide.split(';')]
                            dot_list=[float(x) for x in str(dot).split(';')]
                            run_records.append(dict(excel_row=int(idx)+4,heart=run//2+1,
                                technical_replicate=run%2+1,peptide=peptide,
                                peptides=peptide_list,all_peptides_contained=all(x in row['sequence'] for x in peptide_list),
                                all_dot_products_pass_source_0_7=all(x>=.7 for x in dot_list),
                                dot_product=safe(dot),total_area=safe(area),retention_time=safe(retention)))
                iv=primers.loc[(primers['ORF_id_tr']==row['orf_id']) &
                    primers['Amino acid sequence sORF'].map(lambda v:clean(v)==row['sequence'])]
                r.update(SRM_target_transition_rows=len(peptide_rows),
                    SRM_target_peptides=sorted(set(peptide_rows['Peptide sequence'])),
                    SRM_full_sequence_result_rows=len(rr),SRM_reported_positive_runs=run_records,
                    SRM_status=('reported detected' if run_records else 'targeted, not reported detected' if len(peptide_rows) and len(rr) else 'unavailable or not targeted'),
                    IVT_mutagenesis_source_rows=[int(i)+2 for i in iv.index])
            matches.append(r)
        roster.append(dict(candidate_key=c.candidate_key,candidate_name=c.candidate_name,
            sequence=c.sequence,source_matches=matches,native_function_label_assigned=False))
    (D/'candidate_native_evidence.json').write_text(json.dumps(roster,indent=2)+'\n')
    gb=(D/'NR_024451_1.gb').read_text()
    assert re.search(r'VERSION\s+NR_024451\.1',gb)
    nt=''.join(re.findall('[a-z]+',gb.split('ORIGIN',1)[1].split('//')[0])).upper()
    code=dict(zip((''.join(x) for x in product('TCAG',repeat=3)),
        'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
    peptide=candidates.loc[candidates.candidate_name.eq('JHDM1D-AS1'),'sequence'].iloc[0]
    hits=[]
    for f in range(3):
        translated=''.join(code[nt[i:i+3]] for i in range(f,len(nt)-2,3))
        for m in re.finditer(re.escape(peptide),translated):
            start=f+3*m.start();end=start+3*len(peptide)
            hits.append(dict(start_1based=start+1,last_sense_nt_1based=end,
                stop_start_1based=end+1,stop_end_1based=end+3,frame_offset=f,
                first_codon=nt[start:start+3],stop_codon=nt[end:end+3],
                exact_ATG_to_stop=nt[start:start+3]=='ATG' and nt[end:end+3] in ['TAA','TGA','TAG']))
    linked=next(r for r in roster if r['candidate_name']=='JHDM1D-AS1')
    edits=pd.read_parquet(BRIDGE/'exact_mutation_links.parquet')
    tiles=pd.read_csv(BRIDGE/'motif_associated_tile_edit_context.csv',keep_default_na=False)
    regions=[]
    for source in linked['source_matches']:
        for signature in source.get('SRM_target_peptides',[]):
            at=peptide.find(signature)
            assert at>=0
            for _,edit in edits.loc[edits.candidate_name.eq('JHDM1D-AS1')].iterrows():
                changed=edit.protein_sequence_mut[at:at+len(signature)]
                ts=tiles.loc[tiles.candidate_name.eq('JHDM1D-AS1') & tiles.mutation_group.eq(edit.mutation_group)]
                overlaps=[dict(tile_ID=t.PRISMA_tile_ID,interactor=t.source_interactor,
                    overlap_start_1based=max(at+1,int(t.tile_start_1based)),
                    overlap_end_1based=min(at+len(signature),int(t.tile_end_1based)))
                    for _,t in ts.iterrows() if max(at+1,int(t.tile_start_1based))<=min(at+len(signature),int(t.tile_end_1based))]
                regions.append(dict(SRM_target_peptide=signature,start_1based=at+1,end_1based=at+len(signature),
                    reported_detected_in_source=any(signature in x['peptides'] for x in source['SRM_reported_positive_runs']),
                    edit_family=edit.mutation_group,edited_region=changed,
                    signature_sequence_preserved=signature==changed,binding_tile_overlaps=overlaps,
                    edit_effect_on_MS_detection_measured=False))
    ref=dict(accession='NR_024451.1',reference_length_nt=len(nt),peptide_length_aa=len(peptide),
        exact_peptide=peptide,hits=hits,reference_sequence_compatible=True,
        historical_plasmid_sequence_or_translation_verified=False,
        original_heart_assembly_transcript_length_nt=951,reference_vs_heart_isoform_identity_established=False,
        SRM_signature_edit_and_tile_context=regions)
    (D/'JHDM_REFERENCE_COMPATIBILITY.json').write_text(json.dumps(ref,indent=2)+'\n')
    counts=dict(candidates=len(roster),exact_source_candidates=sum(bool(r['source_matches']) for r in roster),
        full_sequence_source_links=sum(len(r['source_matches']) for r in roster),
        S5_candidates=sum(any(x['table']=='S5' for x in r['source_matches']) for r in roster),
        S3_candidates=sum(any(x['table']=='S3' for x in r['source_matches']) for r in roster),
        SRM_reported_detected_candidates=sum(any(x.get('SRM_status')=='reported detected' for x in r['source_matches']) for r in roster),
        SRM_run_sequence_conflicts=sum(not z['all_peptides_contained'] for r in roster for x in r['source_matches'] for z in x.get('SRM_reported_positive_runs',[])),
        SRM_run_threshold_disagreements=sum(not z['all_dot_products_pass_source_0_7'] for r in roster for x in r['source_matches'] for z in x.get('SRM_reported_positive_runs',[])),
        all_candidates_retained=True,assay_scores_refit=False,native_function_demonstrated=False,
        source_sha256={f.name:sha(f) for f in [s3,s5,D/'NR_024451_1.gb',BRIDGE/'candidate_bridge.csv']})
    (D/'NATIVE_EVIDENCE_AUDIT.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps(counts,indent=2))
    print(json.dumps(ref,indent=2))


if __name__=='__main__':main()
