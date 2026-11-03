"""Exact candidate coverage audit; no new predictor fitting or functional labels."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

P = Path(__file__).resolve().parent
WS = P.parent
O = P / 'native_function_bridge'
B = WS / 'stability_benchmark_2026-10-31'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def normalize(value):
    if not isinstance(value, str):
        return None
    s = re.sub(r'\s+', '', value).upper()
    if s.endswith('*'):
        s = s[:-1]
    return s if s and set(s) <= set('ACDEFGHIKLMNPQRSTVWY') else None


def base_id(value):
    m = re.search(r'c(?:\d+|X|Y)(?:riboseqorf|norep)\d+', str(value))
    return m.group() if m else None


def main():
    assert (O / 'BRIDGE_JOIN_PROTOCOL.txt').exists()
    s2 = O / 'TableS2_publisher.xlsx'
    s3 = O / 'TableS3_publisher.xlsx'
    assert hashlib.md5(s2.read_bytes()).hexdigest() == 'd7dd66225929e5c174f6c22fe8b22b9e'
    assert hashlib.md5(s3.read_bytes()).hexdigest() == 'd7cd5553351eeef16bb4aa08eeb45f8c'
    candidates = pd.read_excel(s2, sheet_name='PRISMA candidates summary')
    assert len(candidates) == 60
    candidates['sequence'] = candidates['Peptide Sequence'].map(normalize)
    assert candidates.sequence.notna().all()
    candidates['candidate_key'] = ['Sandmann_S2_row'+str(i+2) for i in range(len(candidates))]
    exp = 'Experiment sORF ID (transcriptID_ORFstart_ORFstop)'
    assert candidates[exp].is_unique
    natural = pd.read_parquet(B / 'natural_assay_records.parquet')
    natural['sequence'] = natural.protein_sequence.map(normalize)
    native = pd.read_parquet(B / 'results/native_GFP_MP.parquet')
    mutations = pd.read_parquet(B / 'primary_mutation_pairs.parquet')
    s3_interactors = pd.read_excel(s3, sheet_name='Microprotein interactors')
    s3_interactors['sequence'] = s3_interactors['sORF peptide sequence'].map(normalize)
    tiles = pd.read_excel(s3, sheet_name='PRISMA synthesized peptides')
    tiles.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in tiles.columns]
    ectopic = pd.read_excel(s2, sheet_name='Ectopic expression & IVT')
    ectopic['sequence'] = ectopic['Peptide Sequence'].map(normalize)
    assert ectopic.sequence.notna().all() and ectopic.sequence.is_unique
    ectopic = ectopic.set_index('sequence')
    current_detection = next(c for c in ectopic.columns if c.startswith('Detection after overexpression (this study)'))
    metadata = []
    links = []
    edit_links = []
    conflicts = []
    for _, c in candidates.iterrows():
        seq = c.sequence
        rows = natural[natural.sequence.eq(seq)]
        prim = native[native.protein_sequence.eq(seq)]
        ints = s3_interactors[s3_interactors.sequence.eq(seq)]
        confirmed_ids = {c[exp], ectopic.loc[seq, exp], *ints[exp].dropna().unique()}
        own_tiles = tiles[tiles[exp].isin(confirmed_ids)]
        # No first-match gene/name join: candidate identity remains the full sequence.
        valid_tiles = own_tiles[own_tiles['Included in analysis based on bait identification'].astype(str).str.upper().eq('TRUE')]
        b_id = base_id(c['Ribo-seq ORF ID'])
        idrows = natural[natural.protein_identifier.eq(b_id)] if b_id else natural.iloc[:0]
        id_conflicts = idrows[~idrows.sequence.eq(seq)]
        for _, a in id_conflicts.iterrows():
            conflicts.append(dict(candidate_key=c.candidate_key, candidate_name=c['Candidate name'],
                riboseq_id_raw=c['Ribo-seq ORF ID'], matched_base_id=b_id,
                candidate_sequence=seq, assay_sequence=a.protein_sequence,
                assay_record_id=a.record_id, source_sheet=a.source_sheet,
                mapping='identifier overlap with different full peptide sequence; not joined'))
        for _, a in rows.iterrows():
            v = a.to_dict()
            v.update(candidate_key=c.candidate_key, candidate_name=c['Candidate name'],
                     exact_full_sequence=True, native_function_label=False)
            links.append(v)
        own_edits = mutations[mutations.protein_sequence_WT.eq(seq)]
        for _, a in own_edits.iterrows():
            v = a.to_dict()
            v.update(candidate_key=c.candidate_key, candidate_name=c['Candidate name'])
            edit_links.append(v)
        metadata.append(dict(candidate_key=c.candidate_key, candidate_name=c['Candidate name'],
            experiment_orf_id=c[exp], riboseq_id_raw=c['Ribo-seq ORF ID'],
            gene_symbol=c['Gene Symbol'], gene_id=c['Ensembl Gene ID'],
            raw_sequence=c['Peptide Sequence'], sequence=seq, length=len(seq),
            conservation=c['Conservation ORF'], evolution_mode=c['modes'],
            orf_biotype=c['ORF biotype'], source_S2_excel_row=int(c.name)+2,
            exact_natural_assay_records=len(rows), exact_primary_GFP_records=len(prim),
            exact_GFP_records=len(rows[rows.source_sheet.eq('GFP_MP')]),
            exact_mCherry_records=len(rows[rows.source_sheet.eq('MP_mCherry')]),
            exact_mutagenesis_WT_records=len(rows[rows.source_sheet.eq('GFP_MP_mutagenesis')]),
            exact_measured_edit_pairs=len(own_edits), identifier_sequence_conflicts=len(id_conflicts),
            PRISMA_interactor_rows=len(ints), PRISMA_unique_interactor_symbols=ints['Interactor (gene symbol)'].nunique(),
            PRISMA_synthesized_tiles=len(own_tiles), PRISMA_bait_QC_TRUE_tiles=len(valid_tiles),
            PRISMA_tiles_contained_in_candidate=int(own_tiles['Tile peptide sequence'].map(lambda v: isinstance(v, str) and v in seq).sum()),
            source_current_study_tagged_detection=ectopic.loc[seq, current_detection],
            source_summary_vs_ectopic_experiment_ID_agrees=c[exp] == ectopic.loc[seq, exp],
            source_number_MS_studies=c['Number of MS studies (total)'],
            source_number_Ribo_studies=c['Number of Ribo-seq studies'],
            source_HEK293T_RNA_expressed=c['Expressed in HEK293T'],
            native_function_label_assigned=False))
    out = pd.DataFrame(metadata)
    out.to_parquet(O / 'candidate_bridge.parquet', index=False)
    out.to_csv(O / 'candidate_bridge.csv', index=False)
    linkframe = pd.DataFrame(links)
    linkframe.to_parquet(O / 'exact_assay_links.parquet', index=False)
    edits = pd.DataFrame(edit_links)
    edits.to_parquet(O / 'exact_mutation_links.parquet', index=False)
    conflictframe = pd.DataFrame(conflicts)
    conflictframe.to_parquet(O / 'candidate_identifier_conflicts.parquet', index=False)
    candidate_snapshot = candidates.copy()
    candidate_snapshot.to_json(O / 'source_S2_candidates.json', orient='records', indent=2)
    source_files = [s2, s3, B / 'natural_assay_records.parquet', B / 'results/native_GFP_MP.parquet',
                    B / 'primary_mutation_pairs.parquet', O / 'BRIDGE_JOIN_PROTOCOL.txt']
    summary = dict(status='descriptive exact-identity candidate audit completed; independent verification pending',
        candidates=60, unique_candidate_sequences=out.sequence.nunique(),
        candidate_name_duplicates=int(out.candidate_name.duplicated(keep=False).sum()),
        candidate_sequences_with_natural_assay=int(out.exact_natural_assay_records.gt(0).sum()),
        candidate_sequences_with_primary_GFP=int(out.exact_primary_GFP_records.gt(0).sum()),
        candidate_sequences_with_measured_edits=int(out.exact_measured_edit_pairs.gt(0).sum()),
        exact_natural_record_links=len(linkframe), exact_edit_links=len(edits),
        identifier_sequence_conflict_candidates=int(out.identifier_sequence_conflicts.gt(0).sum()),
        identifier_sequence_conflict_links=len(conflictframe),
        all_candidates_retained=True, sequence_normalization='whitespace and one terminal stop only',
        outcome_fitting=False, inferential_testing=False, native_function_label_assigned=False,
        sources={str(f.relative_to(WS)): dict(sha256=sha(f), bytes=f.stat().st_size) for f in source_files})
    (O / 'BRIDGE_AUDIT.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'sources'}, indent=2))
    display = out[out.exact_natural_assay_records.gt(0)]
    print(display[['candidate_name','length','exact_natural_assay_records','exact_primary_GFP_records','exact_measured_edit_pairs','PRISMA_unique_interactor_symbols','source_current_study_tagged_detection']].to_string(index=False))


if __name__ == '__main__':
    main()
