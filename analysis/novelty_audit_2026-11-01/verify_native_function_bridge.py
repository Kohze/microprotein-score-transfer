"""Verify candidate joins from original workbooks using an independent reader."""
from pathlib import Path
import hashlib
import json
from collections import Counter
import openpyxl
import pandas as pd

P = Path(__file__).resolve().parent
WS = P.parent
O = P / 'native_function_bridge'
B = WS / 'stability_benchmark_2026-10-31'


def rows(book, sheet):
    data = book[sheet].iter_rows(values_only=True)
    header = next(data)
    return [dict(zip(header, r)) for r in data if any(v is not None for v in r)]


def peptide(value):
    cleaned = ''.join(value.split()).upper()
    if cleaned[-1:] == '*':
        cleaned = cleaned[:-1]
    assert all(x in 'ACDEFGHIKLMNPQRSTVWY' for x in cleaned)
    return cleaned


def main():
    result = pd.read_parquet(O / 'candidate_bridge.parquet')
    wb = openpyxl.load_workbook(O / 'TableS2_publisher.xlsx', read_only=True, data_only=True)
    original = rows(wb, 'PRISMA candidates summary')
    assert len(original) == len(result) == 60
    guo_path = B / 'inputs/sources/Guo2026_mmc4.xlsx'
    guo = openpyxl.load_workbook(guo_path, read_only=True, data_only=True)
    source = {s: rows(guo, s) for s in ['GFP_MP', 'MP_mCherry', 'GFP_MP_MG132_MLN7243', 'GFP_MP_mutagenesis']}
    natural = pd.read_parquet(B / 'natural_assay_records.parquet')
    # Direct original source row/sequence/value checks for every linked record.
    linked = pd.read_parquet(O / 'exact_assay_links.parquet')
    for _, a in linked.iterrows():
        raw = source[a.source_sheet][int(a.source_excel_row)-2]
        assert raw['protein_identifier'] == a.protein_identifier
        assert raw['protein_sequence'] == a.protein_sequence
        for col in ['stability_r1', 'stability_r2', 'mean_stability', 'NC_stability', 'MG132_stability', 'MLN7243_stability']:
            if col in raw and raw[col] is not None:
                assert abs(float(raw[col])-float(a[col])) < 1e-12
    primary_sequences = set(pd.read_parquet(B / 'results/native_GFP_MP.parquet').protein_sequence)
    edits = pd.read_parquet(B / 'primary_mutation_pairs.parquet')
    source_edits = source['GFP_MP_mutagenesis']
    edit_links = pd.read_parquet(O / 'exact_mutation_links.parquet')
    for i, raw in enumerate(original):
        out = result.iloc[i]
        seq = peptide(raw['Peptide Sequence'])
        assert out.source_S2_excel_row == i+2 and out.sequence == seq
        assert out.candidate_name == raw['Candidate name']
        actual = natural.protein_sequence.eq(seq)
        assert int(actual.sum()) == out.exact_natural_assay_records
        assert int(seq in primary_sequences) == out.exact_primary_GFP_records
        expected_edits = edits.protein_sequence_WT.eq(seq).sum()
        assert expected_edits == out.exact_measured_edit_pairs
    for _, a in edit_links.iterrows():
        wt = [r for r in source_edits if r['protein_identifier'] == a.protein_identifier
              and r['protein_sequence'] == a.protein_sequence_WT and r['group'] == a.group_WT]
        mut = [r for r in source_edits if r['protein_identifier'] == a.protein_identifier
               and r['protein_sequence'] == a.protein_sequence_mut and r['group'] == a.group_mut]
        assert len(wt) == len(mut) == 1
        for ending, row in [('WT', wt[0]), ('mut', mut[0])]:
            for col in ['stability_r1', 'stability_r2', 'mean_stability']:
                assert abs(row[col]-a[col+'_'+ending]) < 1e-12
    # Binding tile positions and edit identities from the raw source table.
    s3 = openpyxl.load_workbook(O / 'TableS3_publisher.xlsx', read_only=True, data_only=True)
    motif_rows = rows(s3, 'Domain-SLiM-matches')
    contexts = pd.read_parquet(O / 'motif_associated_tile_edit_context.parquet')
    for _, a in contexts.iterrows():
        raw = motif_rows[int(a.source_S3_motif_excel_row)-2]
        assert raw['ORF peptide sequence'] == a.WT_sequence
        assert raw['Tile peptide sequence'] == a.WT_tile
        assert raw['Interactor (gene symbol)'] == a.source_interactor
        start, end = int(a.tile_start_1based)-1, int(a.tile_end_1based)
        assert a.WT_sequence[start:end] == a.WT_tile
        assert a.edited_sequence[start:end] == a.edited_tile
        assert sum(x != y for x,y in zip(a.WT_tile, a.edited_tile)) == a.changed_residues_in_tile
    expected_links = [(i+2, r['PRISMA Tile ID']) for i,r in enumerate(motif_rows)
                      if r['ORF peptide sequence'] in set(result[result.exact_primary_GFP_records.gt(0)].sequence)]
    assert len(expected_links) == 5
    assert len(contexts) == 10
    assert len(edit_links) == 32
    all_edits = pd.read_csv(O / 'all32_reporter_edits_and_binding_coverage.csv')
    assert len(all_edits) == 32
    for _, a in all_edits.iterrows():
        assert bool(a.passes_source_Leu_to_Ala_terminal_filter) == ('L' not in a.protein_sequence_WT[-2:])
        assert abs(a.observed_reporter_delta-(a.mean_stability_mut-a.mean_stability_WT)) < 1e-12
    assert ((all_edits.mutation_group == 'L_to_A') & all_edits.passes_source_Leu_to_Ala_terminal_filter).sum() == 12
    receipt = dict(status='passed', original_S2_candidates_checked=60,
        exact_original_Guo_record_links_checked=len(linked), original_Guo_edit_pairs_checked=len(edit_links),
        source_S3_motif_rows_checked=len(expected_links), tile_edit_annotations_checked=len(contexts),
        source_Leu_to_Ala_terminal_filter_checked_for_all32=True,
        alternate_reader='openpyxl vs pandas extraction in analysis',
        source_workbook_sha256=hashlib.sha256(guo_path.read_bytes()).hexdigest(),
        biological_activity_or_causal_effect_verified=False)
    (O / 'INDEPENDENT_BRIDGE_VERIFICATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
