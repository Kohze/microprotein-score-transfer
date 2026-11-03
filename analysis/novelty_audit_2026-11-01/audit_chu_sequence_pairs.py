"""Outcome-free construct/membership audit of the independent C-terminal screen."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook

P=Path(__file__).resolve().parent
WS=P.parent
D=P/'independent_data/Chu2026_supplements'
O=P/'independent_interventions'
O.mkdir(exist_ok=True)


def main():
    file=D/'41467_2026_68979_MOESM1_ESM.xlsx'
    sheet=load_workbook(file,read_only=True,data_only=True).active
    rows=[]
    for i,values in enumerate(sheet.iter_rows(min_row=5,values_only=True),start=5):
        # Only identity/design fields; the PSI/count columns are not accessed.
        gene,transcript,dna,protein,source,length,wt,shared,extension=values[:9]
        if not isinstance(wt,str) or not isinstance(extension,str):
            continue
        # If the RT segment fills the 58-mer there is no retained WT part;
        # Excel represents that empty sequence as a blank cell.
        shared='' if shared is None else shared
        if not isinstance(shared,str):
            continue
        wt,shared,extension=wt.strip(),shared.strip(),extension.strip()
        variant=shared+extension
        rows.append(dict(source_excel_row=i,gene_name=str(gene) if gene is not None else '',
            gene_name_original_type=type(gene).__name__,transcript_id=str(transcript) if transcript is not None else '',
            dna_mutation=str(dna),protein_mutation=str(protein),mutation_source=str(source),
            reported_RT_length=length,wt_sequence=wt,shared_WT_part=shared,
            RT_part=extension,variant_sequence=variant,wt_length=len(wt),variant_length=len(variant),
            wt_endswith_shared=wt.endswith(shared),RT_part_length=len(extension),shared_length=len(shared),
            standard_sequences=set(wt+variant)<=set('ACDEFGHIKLMNPQRSTVWY'),
            simple_append=variant.startswith(wt),identical_pair=variant==wt))
    d=pd.DataFrame(rows)
    d['reported_RT_length']=pd.to_numeric(d.reported_RT_length,errors='coerce')
    canonfile=WS/'thesis_reanalysis_2026-10-31/canonical_all_valid_isoforms.parquet'
    canon=pd.read_parquet(canonfile,columns=['id','gene_id','gene_name','sequence'])
    txmap=defaultdict(set)
    names=defaultdict(set)
    seqmap=defaultdict(set)
    tailgenes=defaultdict(set)
    for row in canon.itertuples():
        tx=str(row.id).split('.')[0]
        gene=str(row.gene_id).split('.')[0]
        txmap[tx].add(gene)
        names[str(row.gene_name)].add(gene)
        seqmap[tx].add(row.sequence[-58:])
        tailgenes[row.sequence[-58:]].add(gene)
    ids=[]
    matched=[]
    mapping_basis=[]
    for row in d.itertuples():
        tx=str(row.transcript_id).split('.')[0]
        genes=txmap.get(tx,set())
        basis='transcript_id'
        if not genes:
            genes=names.get(str(row.gene_name),set())
            basis='gene_symbol'
        if not genes:
            genes=tailgenes.get(row.wt_sequence,set())
            basis='exact_current_C58_sequence'
        ids.append(';'.join(sorted(genes)))
        matched.append(row.wt_sequence in seqmap.get(tx,set()))
        mapping_basis.append(basis if genes else 'unresolved')
    d['mapped_gene_ids']=ids
    d['gene_mapping_basis']=mapping_basis
    d['unique_gene_mapping']=d.mapped_gene_ids.str.startswith('ENSG')&~d.mapped_gene_ids.str.contains(';')
    d['WT_matches_current_transcript_C58']=matched
    source=pd.read_parquet(WS/'stability_benchmark_2026-10-31/sequence_dependence_groups.parquet')
    src=source[source.sources.str.contains('Guo|annotation_reference',regex=True)]
    srcgenes={g.split('.')[0] for val in src.genes.fillna('') for g in val.split(';') if g}
    source_meta=pd.read_parquet(WS/'stability_benchmark_2026-10-31/source_sequence_metadata.parquet')
    srcnames=set(source_meta.gene_name.dropna().astype(str))
    d['source_gene_overlap']=[bool(set(ids.split(';'))&srcgenes) or name in srcnames
                              for ids,name in zip(d.mapped_gene_ids,d.gene_name)]
    # PAP coefficient training is a separate possible source of exposure.
    paptrain=pd.read_csv(P/'independent_data/Voutsinos_degron_train.csv.gz',usecols=['name','aa','set'])
    train_gene=set(paptrain['name'].str.split('_').str[0])
    train30=set(paptrain.aa)
    train25={s[:25] for s in train30}
    d['PAP_training_gene_overlap']=[bool(set(ids.split(';'))&train_gene) for ids in d.mapped_gene_ids]
    def exact_windows(s,size,reference):
        return any(s[i:i+size] in reference for i in range(len(s)-size+1))
    d['PAP_training_exact30_window']=[exact_windows(a,30,train30) or exact_windows(b,30,train30)
                                      for a,b in zip(d.wt_sequence,d.variant_sequence)]
    d['human25_training_exact25_window']=[exact_windows(a,25,train25) or exact_windows(b,25,train25)
                                          for a,b in zip(d.wt_sequence,d.variant_sequence)]
    d['reported_RT_length_consistent']=d.RT_part_length.eq(d.reported_RT_length.clip(upper=58))
    d.to_parquet(O/'sequence_pair_audit.parquet',index=False)
    d.to_csv(O/'sequence_pair_audit.csv',index=False)
    counts=dict(rows=len(d),gene_names=int(d.gene_name.nunique()),unique_wt_sequences=int(d.wt_sequence.nunique()),
        unique_variant_sequences=int(d.variant_sequence.nunique()),wt_length_counts=d.wt_length.value_counts().to_dict(),
        variant_length_counts=d.variant_length.value_counts().to_dict(),standard_pairs=int(d.standard_sequences.sum()),
        WT_endswith_shared=int(d.wt_endswith_shared.sum()),simple_append_pairs=int(d.simple_append.sum()),
        identical_pairs=int(d.identical_pair.sum()),unique_gene_mapping=int(d.unique_gene_mapping.sum()),
        mapping_basis_counts=d.gene_mapping_basis.value_counts().to_dict(),
        reported_RT_length_consistent=int(d.reported_RT_length_consistent.sum()),
        current_transcript_C58_matches=int(d.WT_matches_current_transcript_C58.sum()),
        source_gene_overlap=int(d.source_gene_overlap.sum()),PAP_training_gene_overlap=int(d.PAP_training_gene_overlap.sum()),
        PAP_exact30_window_pairs=int(d.PAP_training_exact30_window.sum()),human25_exact25_window_pairs=int(d.human25_training_exact25_window.sum()))
    receipt=dict(status='outcome-free sequence and identifier audit complete; similarity/measurement audit pending',
        table_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),canonical_reference_sha256=hashlib.sha256(canonfile.read_bytes()).hexdigest(),
        counts=counts,outcome_values_loaded=False,
        construct='Independent GFP reporter with canonical and readthrough-derived C-terminal windows; not generally simple extension of the measured WT cargo',
        primary_gene_identifier='mapped ENSG when unique; source gene symbol overlap adds a conservative exclusion',
        limitations=['Historical ENST/sequence versions may differ from current GENCODE','Gene/PAP overlap flags do not establish all homology independence'])
    (O/'SEQUENCE_PAIR_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
