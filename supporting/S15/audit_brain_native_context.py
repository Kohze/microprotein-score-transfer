"""Exact-sequence selected-panel audit and native-MS endpoint feasibility.

No predictor fitting, score/outcome association or new functional labels.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import openpyxl

P=Path(__file__).resolve().parent
D=P/'native_detection_prior'
F=D/'BrainAtlas2026/43587_2026_1207_MOESM4_ESM.xlsx'
ROSTER=P/'native_system_prior/candidate_primaries/candidate_native_evidence.json'


def read_rows(workbook,sheet):
    s=workbook[sheet]
    # The source declares A1:A1 dimensions on multiple populated worksheets.
    # Reset dimensions to read physical XML rows; do not edit the workbook.
    s.reset_dimensions()
    rows=iter(s.values);next(rows);next(rows);h=next(rows)
    return [dict((key,row[i] if i<len(row) else None) for i,key in enumerate(h) if key is not None)
            for row in rows if row and row[0] is not None]


def main():
    w=openpyxl.load_workbook(F,read_only=True,data_only=True)
    specs={'S1 Discovery':'Microprotein Sequence','S2 Proteomics':'sequence',
           'S3 Tryptic Peptides':'sequence','S5 ShortStop':'Microprotein Sequence'}
    tables={name:read_rows(w,name) for name in specs}
    w.close()
    roster=json.loads(ROSTER.read_text(encoding='utf-8'))
    output=[]
    for c in roster:
        output.append(dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],
            sequence=c['sequence'],source_rows={name:[r for r in tables[name] if r[col]==c['sequence']]
                                              for name,col in specs.items()}))
    s1=tables['S1 Discovery'];s2=tables['S2 Proteomics'];s3=tables['S3 Tryptic Peptides']
    unreviewed=[r for r in s2 if r['Status']=='Unreviewed']
    count_positive=[r for r in unreviewed if r['total_unique_spectral_counts']>0]
    case63=next(c for c in output if c['candidate_key']=='Sandmann_S2_row28')
    case50=next(c for c in output if c['candidate_key']=='Sandmann_S2_row27')
    rp11=next(c for c in output if c['candidate_name']=='RP11-620J15.3')
    assert len(case63['sequence'])==63 and len(case50['sequence'])==50
    assert case63['source_rows']['S1 Discovery'][0]['DDA Grade']=='Strong'
    assert case63['source_rows']['S2 Proteomics'][0]['total_unique_spectral_counts']==1004
    assert case50['source_rows']['S1 Discovery'][0]['MS Evidence Type']=='No MS'
    assert rp11['source_rows']['S1 Discovery'][0]['MS Evidence Type']=='No MS'
    audit=dict(scope='Selected exact-candidate lookup and source endpoint feasibility only',
        source=dict(path=str(F.relative_to(P.parent)),sha256=hashlib.sha256(F.read_bytes()).hexdigest()),
        roster_source=dict(path=str(ROSTER.relative_to(P.parent)),sha256=hashlib.sha256(ROSTER.read_bytes()).hexdigest()),
        normalization='No sequence normalization at this join; exact full sequence from the previously verified roster',
        worksheet_reading='openpyxl read_only with reset_dimensions; source workbook unchanged',
        table_rows={k:len(v) for k,v in tables.items()},
        exact_candidate_matches={k:sum(bool(c['source_rows'][k]) for c in output) for k in specs},
        S1_annotation_status=dict(Counter(r['Annotation Status'] for r in s1)),
        S1_MS_evidence_type=dict(Counter(r['MS Evidence Type'] for r in s1)),
        S1_DDA_grade=dict(Counter(r['DDA Grade'] for r in s1)),
        S2_status=dict(Counter(r['Status'] for r in s2)),
        unreviewed_S2_rows=len(unreviewed),unreviewed_with_positive_unique_PSM_counts=len(count_positive),
        unreviewed_zero_unique_counts=sum(r['total_unique_spectral_counts']==0 for r in unreviewed),
        exact_S3_coverage_of_count_positive=sum(any(q['sequence']==r['sequence'] for q in s3) for r in count_positive),
        count_positive_standard_AA_31_to_150=sum(31<=len(r['sequence'])<=150 and set(r['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') for r in count_positive),
        outcomes_not_established=['Native half-life','Molar abundance across products','Peptide-dependent activity','Functional-negative labels'],
        target_notes=['TMT disease log fold changes are within-product relative responses, not baseline between-product concentrations',
                      'Unique PSM counts measure MS recovery and depend on length, tryptic capacity, RNA/translation, sequence and search workflow',
                      'RiboCode_SAM no-MS subset is ShortStop-selected and cannot serve as an unselected independent ShortStop validation cohort',
                      'DDA search and PROSIT grading are not described as requiring ShortStop positivity',
                      'No-MS or absent lookup is not a functional negative; no source result is reclassified here'])
    for name,obj in [('brain_candidate_sequence_evidence.json',output),('BRAIN_NATIVE_CONTEXT_AUDIT.json',audit)]:
        (D/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ['table_rows','exact_candidate_matches','S1_annotation_status','S1_MS_evidence_type','S1_DDA_grade','unreviewed_with_positive_unique_PSM_counts','exact_S3_coverage_of_count_positive','count_positive_standard_AA_31_to_150']},indent=2))


if __name__=='__main__':main()
