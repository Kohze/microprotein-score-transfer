"""Compare sequence-only inputs with the22,564-row C23 development screen."""
from pathlib import Path
import hashlib
import json
import datetime
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'

def main():
    (O/'EXPOSURE_PROTOCOL.txt').write_text('''TERMINAL-MODEL DEVELOPMENT-SCREEN EXPOSURE AUDIT
This is a secondary provenance audit after other predictor outcomes are known.
No DEGRONOPEDIA versus target associations have been computed. Pinned weights
reproduce all12 author examples. The original model has2997 named features;
the columnSet regex is not invoked by its inference path, so it is not a
Whole/Ogryzek4-only model. Preserve all original descriptor functions.

Hickey2020/2021 Supplementary Table5A explicitly republishes columnsA-L from
Koren2018 TableS7A. Its22564 rows match the DEGRONOPEDIA C23 development-screen
count. Treat it as a public primary redistribution of that screen, not the
unavailable row-specific90/10 split. Inspect sequence/label validity and record
both exact C23 matches and available exact gene-symbol overlap. An exact match
proves development-screen membership, not membership of the90% training split.
Absence of a symbol match does not prove absence of alias/gene exposure.

Apply the frozen Chu1743 common eligibility and native14143 tile eligibility.
Report WT/RT endpoints separately and their union. Identify neither-endpoint
screen matches and a stricter available-symbol-disjoint sensitivity. No target
scores, model direction or significance values choose these membership rules.
Do not call unfiltered model comparisons equally independent. N-model training
screens are separate and remain unaudited; initiator processing is not assumed.
''',encoding='utf-8')
    file=O/'Hickey2021_STable5.xlsx'
    d=pd.read_excel(file,sheet_name='(A) All C-terminal 23mers',header=4)
    assert len(d)==22564
    # These two sequence columns are a useful internal transcription check.
    seq=d['Peptide amino acid sequence'].astype(str).str.removesuffix('*')
    assert seq.equals(d['Analysed Sequence'].astype(str))
    d['C23_sequence']=seq
    d['sequence_valid']=seq.str.len().eq(23)&seq.map(lambda s:set(s)<=set('ACDEFGHIKLMNPQRSTVWY'))
    d['PSI_numeric']=pd.to_numeric(d.PSI,errors='coerce')
    date_cells=[dict(source_excel_row=int(i)+6,value=str(v)) for i,v in d.Gene_ID.items()
        if isinstance(v,(datetime.datetime,datetime.date))]
    export=d[['Transcript_ID','Gene_ID','C23_sequence','sequence_valid','PSI_numeric']].copy()
    export.insert(0,'source_excel_row',np.arange(len(d))+6)
    for column in ['Transcript_ID','Gene_ID']:export[column]=export[column].astype('string')
    export.to_parquet(O/'Koren2018_C23_republished_source.parquet',index=False)
    known=set(d.loc[d.sequence_valid,'C23_sequence'])
    symbols=set(d.Gene_ID.dropna().astype(str).str.upper())
    pairs=pd.read_parquet(P/'independent_interventions/evaluated_predictions.parquet')
    out=pairs[['source_excel_row','wt_sequence','variant_sequence','primary60_common_eligible','target_dependence_group']].copy()
    # Audit available symbol fields without inventing an identifier mapping.
    symbol_columns=[c for c in pairs if ('gene' in c.lower() and ('symbol' in c.lower() or c.lower() in ['gene','gene_name']))]
    for label,column in [('WT','wt_sequence'),('RT','variant_sequence')]:
        out[label+'_C23']=pairs[column].str[-23:]
        out[label+'_exact_development_screen_match']=out[label+'_C23'].isin(known)
    out['either_endpoint_screen_match']=out.WT_exact_development_screen_match|out.RT_exact_development_screen_match
    out['available_gene_symbol_screen_overlap']=False
    for column in symbol_columns:
        out['available_gene_symbol_screen_overlap']|=pairs[column].astype(str).str.upper().isin(symbols)
        out[column]=pairs[column]
    out.to_parquet(O/'Chu_C23_exposure_audit.parquet',index=False)
    out.to_csv(O/'Chu_C23_exposure_audit.csv',index=False)
    native=pd.read_parquet(P/'independent_test/evaluated_external_predictions.parquet',columns=['aa','primary60_eligible'])
    native['C23']=native.aa.str[-23:]
    native['exact_development_screen_match']=native.C23.isin(known)
    native.to_parquet(O/'Voutsinos_C23_exposure_audit.parquet',index=False)
    z=out[out.primary60_common_eligible]
    result=dict(source='Hickey2020/2021 SupplementalTable5A republication of Koren2018TableS7A columnsA-L',
        source_doi='10.25386/genetics.13273286',source_file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        screen_rows=len(d),valid_standard_C23=int(d.sequence_valid.sum()),unique_standard_C23=len(known),
        missing_PSI=int(d.PSI_numeric.isna().sum()),C23_length_counts={str(k):int(v) for k,v in seq.str.len().value_counts().items()},
        available_gene_symbols=len(symbols),audit_symbol_columns=symbol_columns,
        original_gene_ID_date_artifacts=date_cells,
        primary_pairs=len(z),WT_screen_matches=int(z.WT_exact_development_screen_match.sum()),
        RT_screen_matches=int(z.RT_exact_development_screen_match.sum()),either_endpoint_screen_matches=int(z.either_endpoint_screen_match.sum()),
        neither_endpoint_exact_screen_match=int((~z.either_endpoint_screen_match).sum()),
        available_symbol_overlap=int(z.available_gene_symbol_screen_overlap.sum()),
        neither_endpoint_or_symbol_overlap=int((~z.either_endpoint_screen_match&~z.available_gene_symbol_screen_overlap).sum()),
        native_primary_tiles=int(native.primary60_eligible.sum()),
        native_C23_screen_matches=int((native.primary60_eligible&native.exact_development_screen_match).sum()),
        model_training_split_membership_available=False,target_model_associations_computed=False,
        limit='Exact screen/available-symbol matches only; no certified individual-training exclusion, exhaustive aliases or homology exclusion')
    (O/'SCREEN_EXPOSURE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
