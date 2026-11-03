"""Post-hoc annotation of existing edits against source-supported WT binding tiles."""
from pathlib import Path
import json
import pandas as pd

P = Path(__file__).resolve().parent
O = P / 'native_function_bridge'


def occurrences(sequence, tile):
    return [i for i in range(len(sequence)-len(tile)+1) if sequence[i:i+len(tile)] == tile]


def main():
    (O / 'BINDING_TILE_EDIT_ADDENDUM.txt').write_text('''POST-HOC WT BINDING-TILE / EXISTING EDIT ANNOTATION
The exact overlap counts and six reporter effects for three candidates have
already been inspected. Retain every one of the 32 available edit pairs.
For each source domain-SLiM row, require exact full WT identity and locate
all exact occurrences of its 15mer binding tile in that WT. Global Leu-to-Ala
changes mapped leucines. The verified append retains the entire WT sequence.
Record residue changes inside a motif-associated binding tile, not a claimed
change to the causal motif or actual binding. No edited PRISMA or native
functional measurement exists here. Do not infer kinase causality from a
source domain/SLiM match or claim that phosphorylation was changed by the edit.
This is candidate design support, without enrichment or significance testing.
''', encoding='utf-8')
    candidates = pd.read_parquet(O / 'candidate_bridge.parquet')
    edits = pd.read_parquet(O / 'exact_mutation_links.parquet')
    motifs = pd.read_excel(O / 'TableS3_publisher.xlsx', sheet_name='Domain-SLiM-matches')
    phospho = pd.read_excel(O / 'TableS3_publisher.xlsx', sheet_name='Phosphorylated microproteins')
    annotations = []
    for _, e in edits.iterrows():
        c = candidates[candidates.candidate_key.eq(e.candidate_key)].iloc[0]
        seq = e.protein_sequence_WT
        target = e.protein_sequence_mut
        if e.mutation_group == 'L_to_A':
            assert target == seq.replace('L', 'A')
        else:
            assert e.mutation_group == 'microprotein_10mer_ext' and target == seq+'DNYNEPKANQ'
        rows = motifs[motifs['ORF peptide sequence'].eq(seq)]
        for source_idx, t in rows.iterrows():
            tile = t['Tile peptide sequence']
            positions = occurrences(seq, tile)
            assert positions
            for start in positions:
                changed = [i+1 for i in range(start, start+len(tile)) if target[i] != seq[i]]
                annotations.append(dict(candidate_key=e.candidate_key, candidate_name=e.candidate_name,
                    gene_symbol=c.gene_symbol, protein_identifier=e.protein_identifier,
                    mutation_group=e.mutation_group, WT_sequence=seq, edited_sequence=target,
                    source_S3_motif_excel_row=int(source_idx)+2,
                    PRISMA_tile_ID=t['PRISMA Tile ID'], WT_tile=tile,
                    tile_start_1based=start+1, tile_end_1based=start+len(tile),
                    edited_tile=target[start:start+len(tile)], changed_residues_in_tile=len(changed),
                    changed_WT_positions_1based=','.join(map(str, changed)),
                    source_interactor=t['Interactor (gene symbol)'], source_motif_ID=t['SLiM ID of interacting SLiM in ORF'],
                    source_bait_QC=t['Included in analysis based on bait identification'],
                    mean_reporter_WT=e.mean_stability_WT, mean_reporter_edit=e.mean_stability_mut,
                    observed_reporter_delta=e.mean_stability_mut-e.mean_stability_WT,
                    replicate1_delta=e.stability_r1_mut-e.stability_r1_WT,
                    replicate2_delta=e.stability_r2_mut-e.stability_r2_WT,
                    motif_causal_residues_validated=False, edited_binding_measured=False,
                    edited_native_function_measured=False))
    out = pd.DataFrame(annotations)
    out.to_parquet(O / 'motif_associated_tile_edit_context.parquet', index=False)
    out.to_csv(O / 'motif_associated_tile_edit_context.csv', index=False)
    edits['observed_reporter_delta'] = edits.mean_stability_mut-edits.mean_stability_WT
    edits['replicate1_delta'] = edits.stability_r1_mut-edits.stability_r1_WT
    edits['replicate2_delta'] = edits.stability_r2_mut-edits.stability_r2_WT
    edits['WT_has_Leu_in_last2'] = edits.protein_sequence_WT.str[-2:].str.contains('L')
    edits['passes_source_Leu_to_Ala_terminal_filter'] = ~edits.WT_has_Leu_in_last2
    edits['source_domain_SLiM_rows'] = edits.protein_sequence_WT.map(
        lambda s: int(motifs['ORF peptide sequence'].eq(s).sum()))
    edits.to_csv(O / 'all32_reporter_edits_and_binding_coverage.csv', index=False)
    # The phosphorylation data are tagged-expression observations, not activity labels.
    hit = candidates[candidates.exact_primary_GFP_records.gt(0)]
    matched_phospho = phospho[phospho['Experiment sORF ID (transcriptID_ORFstart_ORFstop)'].isin(hit.experiment_orf_id)]
    matched_phospho.to_csv(O / 'source_tagged_phosphorylation_overlap.csv', index=False)
    summary = dict(status='completed; descriptive post-hoc annotation',
        all_available_edit_pairs=len(edits), Leu_to_Ala_pairs=int(edits.mutation_group.eq('L_to_A').sum()),
        append_pairs=int(edits.mutation_group.eq('microprotein_10mer_ext').sum()),
        Leu_to_Ala_pairs_passing_source_terminal_filter=int((edits.mutation_group.eq('L_to_A') & edits.passes_source_Leu_to_Ala_terminal_filter).sum()),
        observed_positive_delta_pairs=int(edits.observed_reporter_delta.gt(0).sum()),
        observed_negative_delta_pairs=int(edits.observed_reporter_delta.lt(0).sum()),
        candidate_sequences_with_source_domain_SLiM_rows=out.candidate_key.nunique(),
        distinct_motif_associated_WT_tiles=out[['candidate_key','PRISMA_tile_ID']].drop_duplicates().shape[0],
        tile_edit_annotations=len(out), source_tagged_phosphorylation_rows=len(matched_phospho),
        source_tagged_phosphorylation_candidate_names=matched_phospho['Candidate name'].unique().tolist(),
        functional_or_binding_loss_demonstrated=False, causal_kinase_assignment=False,
        inferential_testing=False, revised_candidate_hypothesis='Test whether reporter-stabilizing edits preserve the same candidate\'s WT binding and native activity')
    (O / 'EDIT_CONTEXT_AUDIT.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
