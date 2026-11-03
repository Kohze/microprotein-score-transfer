"""Pin public inputs and prepare context-specific, auditable assay records."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import hashlib,json,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
WS=ROOT.parent
DEEP=WS/'deep_novelty_2026-10-31'
BASE=WS/'thesis_reanalysis_2026-10-31'
AA=set('ACDEFGHIKLMNPQRSTVWY')


def main():
    inputs=ROOT/'inputs';inputs.mkdir(exist_ok=True)
    needed=['Guo2026_all_assay_constructs.parquet','Guo2026_within_library_mutant_pairs.parquet',
            'ShortStop_Guo_sequence_checked_mapping.parquet','Yang2024_independent_constructs_and_assays.parquet',
            'ShortStop_training_IDs_and_labels.parquet','PAP_HUMAN30_PARSED_COEFFICIENTS.json',
            'GUO_SUPPLEMENT_AUDIT.json','SHORTSTOP_SEQUENCE_MAPPING_AUDIT.json','EXTERNAL_ENDPOINT_FEASIBILITY.json',
            'sources/Guo2026_mmc2.xlsx','sources/Guo2026_mmc3.xlsx','sources/Guo2026_mmc4.xlsx',
            'sources/Guo2026_mmc7.xlsx','sources/Guo2026_mmc1.pdf',
            'sources/shortstop_v1.2_gencode_smorfs_classifications.csv','sources/ShortStop_v1.2.0_release_metadata.json',
            'sources/Yang2024_mmc11.xlsx','sources/Yang2024_mmc12.xlsx',
            'Voutsinos2026_code/PAP.py','ShortStop2025_code/train_model.py','ShortStop2025_code/consensus_validation.md']
    manifest={}
    for name in needed:
        src=DEEP/name;dest=inputs/name;dest.parent.mkdir(exist_ok=True,parents=True)
        if not dest.exists():shutil.copy2(src,dest)
        sha=hashlib.sha256((src if src.exists() else dest).read_bytes()).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==sha
        manifest[name]=dict(bytes=dest.stat().st_size,sha256=sha)
    for name in ['matched_comprehensive.parquet','norf_catalogue_audit.parquet','canonical_proteins.parquet',
                 'sources/uniprot_reviewed_human.fasta','sources/AAidx_protr_1.7.4.csv']:
        dest=inputs/'reference'/name;dest.parent.mkdir(exist_ok=True,parents=True)
        if not dest.exists():shutil.copy2(BASE/name,dest)
        manifest['reference/'+name]=dict(bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    source=ROOT/'inputs/Guo2026_all_assay_constructs.parquet'
    a=pd.read_parquet(source)
    natural=a.loc[a.group.isin(['microprotein_WT','microprotein_ms_HLA','microprotein_ms_nonHLA']) & a.sequence_valid].copy()
    byid=natural.groupby('protein_identifier').protein_sequence.nunique()
    conflict_ids=set(byid[byid.gt(1)].index)
    natural['inconsistent_parent_id']=natural.protein_identifier.isin(conflict_ids)
    natural['cohort']=np.where(natural.group.eq('microprotein_WT'),'WT',natural.group)
    natural['record_id']=natural.source_sheet+'::'+natural.group+'::'+natural.protein_identifier
    assert not natural.record_id.duplicated().any()
    meta=pd.read_excel(inputs/'sources/Guo2026_mmc2.xlsx',sheet_name='Sheet1')
    meta=meta.rename(columns={'protein_sequence':'sequence','orf_biotype':'source_biotype'})
    assert not meta.sequence.duplicated().any()
    cols=['sequence','source_biotype','gene_id','transcript_id','gene_name','nt_sequence','orf_name',
          'average disorder score(SPOT-Disorder)','hydrophobicity']
    meta[cols].to_parquet(ROOT/'source_sequence_metadata.parquet',index=False)
    n=pd.read_parquet(inputs/'reference/norf_catalogue_audit.parquet')
    can=pd.read_parquet(inputs/'reference/canonical_proteins.parquet')
    seq_genes=defaultdict(set);seq_groups=defaultdict(set);seq_types=defaultdict(set)
    for row in n.itertuples():
        seq_groups[row.sequence].add(str(row.group))
        if isinstance(row.all_genes,str):
            seq_genes[row.sequence].update(x.split('.')[0] for x in row.all_genes.split(';') if x)
        seq_types[row.sequence].add(str(row.orf_type))
    for row in can.itertuples():
        seq_groups[row.sequence].add(str(row.group));seq_genes[row.sequence].add(str(row.gene_id).split('.')[0])
    for row in meta.itertuples():
        if isinstance(row.gene_id,str):seq_genes[row.sequence].update(x.split('.')[0] for x in row.gene_id.split(';') if x)
    natural['genes']=natural.protein_sequence.map(lambda s:';'.join(sorted(seq_genes[s])))
    natural['catalogue_groups']=natural.protein_sequence.map(lambda s:';'.join(sorted(seq_groups[s])))
    natural['orf_type']=natural.protein_sequence.map(lambda s:';'.join(sorted(seq_types[s])))
    natural=natural.merge(meta[['sequence','source_biotype']],left_on='protein_sequence',right_on='sequence',how='left',validate='many_to_one').drop(columns='sequence')
    natural.to_parquet(ROOT/'natural_assay_records.parquet',index=False)
    yang=pd.read_parquet(inputs/'Yang2024_independent_constructs_and_assays.parquet')
    yang['genes']=yang.native_sequence.map(lambda s:';'.join(sorted(seq_genes[s])))
    yang['catalogue_groups']=yang.native_sequence.map(lambda s:';'.join(sorted(seq_groups[s])))
    yang.to_parquet(ROOT/'validation_assay_records.parquet',index=False)
    ref=pd.read_parquet(inputs/'reference/matched_comprehensive.parquet')
    ref['genes']=ref.sequence.map(lambda s:';'.join(sorted(seq_genes[s])))
    ref.to_parquet(ROOT/'annotation_reference.parquet',index=False)
    # Current available counterparts of ShortStop's actual supervised positives.
    labels=pd.read_parquet(inputs/'ShortStop_training_IDs_and_labels.parquet')
    positive_ids=set(labels.loc[labels.local.isin(['Cytoplasm','Secreted']),'orf_id'])
    fasta={};ident=None
    for line in (inputs/'reference/sources/uniprot_reviewed_human.fasta').read_text().splitlines():
        if line.startswith('>'):ident=line[1:].split('|')[1];fasta[ident]=''
        elif ident:fasta[ident]+=line.strip()
    positives=pd.DataFrame([dict(id=k,sequence=fasta[k]) for k in sorted(positive_ids) if k in fasta])
    positives.to_parquet(ROOT/'available_shortstop_positive_counterparts.parquet',index=False)
    pairs=pd.read_parquet(inputs/'Guo2026_within_library_mutant_pairs.parquet')
    pairs=pairs.loc[pairs.mutation_group.isin(['L_to_A','microprotein_10mer_ext']) & pairs.pair_sequence_valid & pairs.mutation_verified].copy()
    pairs['inconsistent_parent_id']=pairs.protein_identifier.isin(conflict_ids)
    pairs.to_parquet(ROOT/'primary_mutation_pairs.parquet',index=False)
    # No independent-validation predictor associations are computed here.
    development=natural.loc[natural.source_sheet.eq('GFP_MP')&natural.group.eq('microprotein_WT')&~natural.inconsistent_parent_id]
    both=development[['stability_r1','stability_r2']].dropna()
    disagreement=float(np.median(abs(both.stability_r1-both.stability_r2)))
    receipt=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256((ROOT/'EXECUTION_PROTOCOL.txt').read_bytes()).hexdigest(),
        input_files=manifest,missing_full_STAR_methods=True,
        inconsistent_WT_identifiers=sorted(conflict_ids),natural_records=len(natural),
        cohort_counts=natural.groupby(['source_sheet','group']).size().reset_index(name='n').to_dict('records'),
        primary_development_rows=len(development),primary_development_measured=int(development.mean_stability.notna().sum()),
        native_replicate_median_absolute_disagreement=disagreement,descriptive_MAE_resolution=disagreement/2,
        independent_validation_predictor_associations_computed=False,
        historical_ShortStop_positive_sequence_snapshot_available=False,
        available_ShortStop_positive_counterparts=len(positives))
    (ROOT/'INPUT_FREEZE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='input_files'},indent=2),flush=True)


if __name__=='__main__':main()
