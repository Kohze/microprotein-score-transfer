"""Annotate the frozen roster using exact original Hofman product sequences.

This reads source summaries, not raw Ribo-seq/CRISPR reads or mass spectra.
Only one terminal stop marker is removed from a source peptide string.
"""
from pathlib import Path
from collections import defaultdict, Counter
import csv, json, hashlib
import openpyxl

P = Path(__file__).resolve().parent
S = P / 'detection_evolution_prior/Hofman2024'
W = S / 'repository_supplement'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cleaned(value):
    if not isinstance(value, str):
        return ''
    return value[:-1] if value.endswith('*') else value

def sheet(book, name, header=3):
    ws = book[name]
    rows = list(ws.iter_rows(values_only=True))
    fields = list(rows[header-1])
    return [(i, dict(zip(fields, row)), row) for i, row in enumerate(rows[header:], header+1)
            if any(v is not None for v in row)]

def main():
    roster = P / 'native_function_bridge/candidate_bridge.csv'
    with roster.open(encoding='utf-8-sig', newline='') as f:
        candidates = list(csv.DictReader(f))
    assert len(candidates) == 60
    paths = {n: next(W.glob('*mmc'+str(n)+'.xlsx')) for n in [2,3,6]}
    b1 = openpyxl.load_workbook(paths[2], read_only=True, data_only=True)
    definitions = sheet(b1, 'Table S1P')
    samples = {d['orf_id']: d['number_samples'] for _,d,_ in sheet(b1, 'Table S1N')}
    seq_index = defaultdict(list)
    for row,d,_ in definitions:
        d = dict(d, source_excel_row=row)
        seq_index[cleaned(d['pep'])].append(d)
    assert len(definitions) == 8008
    b1.close()
    b2 = openpyxl.load_workbook(paths[3], read_only=True, data_only=True)
    source_a = sheet(b2, 'Table_S2A')
    screen = {d['Annotation']: dict(d, source_excel_row=row)
              for row,d,_ in sheet(b2, 'Table_S2E')}
    screen_seq_index = defaultdict(dict)
    for row,d,_ in source_a:
        seq = cleaned(d.get('aa sequence'))
        if not seq:
            continue
        name, typ = d['Gene Name'], d['Type']
        key = (name,typ,seq)
        screen_seq_index[seq].setdefault(key, {'name':name,'type':typ,'sequence':seq,'rows':[]})['rows'].append(row)
    classes = Counter()
    tiling = []
    category = None
    for row in b2['Table_S2L'].iter_rows(min_row=5,values_only=True):
        if row[0] is not None:
            category = row[0]
        if isinstance(row[1],str):
            classes[category] += 1
            tiling.append({'category':category,'annotation':row[1]})
    out = []
    exact_details = []
    for c in candidates:
        seq = c['sequence']
        matches = seq_index.get(seq, [])
        definition_links = []
        for d in matches:
            definition_links.append({k:d[k] for k in ['source_excel_row','orf_id','gene','gene_name','orf_biotype','pep']}
                                    | {'samples_PPM_gt1':samples.get(d['orf_id'])})
        screen_links = []
        for d in screen_seq_index.get(seq,{}).values():
            typ = {'uoORF':'overlapping.uORF'}.get(d['type'],d['type'])
            annotation = d['name']+'_'+typ
            e = screen.get(annotation)
            screen_links.append(dict(d, summary_annotation=annotation, summary_found=e is not None,
                                     source_hit=e.get('Hit') if e else None,
                                     source_summary_excel_row=e.get('source_excel_row') if e else None,
                                     best4_scores={k:e[k] for k in e if isinstance(k,str) and 'best_4' in k} if e else {}))
        rec = dict(candidate_key=c['candidate_key'],candidate_name=c['candidate_name'],sequence=seq,
                   length=len(seq),exact_definition_records=len(matches),
                   exact_definition_PPM_gt1_atleast5=sum(isinstance(d['samples_PPM_gt1'],(int,float)) and d['samples_PPM_gt1']>=5 for d in definition_links),
                   exact_screen_products=len(screen_links),
                   exact_screen_resolved_summaries=sum(d['summary_found'] for d in screen_links),
                   exact_screen_unresolved_summaries=sum(not d['summary_found'] for d in screen_links),
                   exact_screen_hit_products=sum(d['source_hit']==1 for d in screen_links),
                   existing_primary_GFP_records=c['exact_primary_GFP_records'],
                   existing_measured_edit_pairs=c['exact_measured_edit_pairs'],
                   source_definitions=json.dumps(definition_links,separators=(',',':')),
                   source_screen_products=json.dumps(screen_links,separators=(',',':')),
                   native_peptide_function_validated_by_this_join=False)
        out.append(rec)
        if matches or screen_links:
            exact_details.append(rec)
    b2.close()
    b5 = openpyxl.load_workbook(paths[6],read_only=True,data_only=True)
    constructs = [{'excel_row':row,'construct':d['Construct name'],'vector':d['Vector'],
                   'clone_sequence':d['Clone Sequence']}
                  for row,d,_ in sheet(b5,'Table S5S')]
    b5.close()
    asdurf = [dict(d,source_excel_row=row) for row,d,_ in source_a
              if d.get('Gene Name')=='ASNSD1' and d.get('Type')=='uORF']
    asdurf_sequences = set(cleaned(d['aa sequence']) for d in asdurf)
    assert len(asdurf_sequences)==1 and len(next(iter(asdurf_sequences)))==96
    with (S/'CANDIDATE_ASSAY_CONTEXT.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    summary = dict(source_paths={str(p):digest(p) for p in paths.values()},roster_sha256=digest(roster),
                   source_definition_records=len(definitions),candidate_records=len(out),
                   exact_definition_candidates=sum(d['exact_definition_records']>0 for d in out),
                   exact_definition_records=sum(d['exact_definition_records'] for d in out),
                   exact_translated_atleast5_candidates=sum(d['exact_definition_PPM_gt1_atleast5']>0 for d in out),
                   exact_screen_candidates=sum(d['exact_screen_products']>0 for d in out),
                   exact_screen_summary_resolved_candidates=sum(d['exact_screen_resolved_summaries']>0 for d in out),
                   exact_screen_summary_unresolved_candidates=sum(d['exact_screen_unresolved_summaries']>0 for d in out),
                   exact_screen_hit_candidates=sum(d['exact_screen_hit_products']>0 for d in out),
                   unresolved_screen_summary_annotations=[d['summary_annotation'] for r in out for d in json.loads(r['source_screen_products']) if not d['summary_found']],
                   tiling_classification_counts=dict(classes),tiling_rows=tiling,ASNSD1_library_exon_records=asdurf,
                   ASNSD1_96aa_sequence=next(iter(asdurf_sequences)),
                   ASNSD1_exact_roster_matches=sum(c['sequence'] in asdurf_sequences for c in candidates),
                   clone_table_rows=constructs,matched_candidate_details=exact_details,
                   new_native_or_edited_function_measurement=False,new_model_fitting_or_inference=False)
    (S/'CANDIDATE_ASSAY_AUDIT.json').write_text(json.dumps(summary,indent=2,default=str)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['clone_table_rows','matched_candidate_details','ASNSD1_definition_records','tiling_rows']},indent=2))
    for d in exact_details:
        print(d['candidate_name'],d['length'],'definitions',d['exact_definition_records'],'translated>=5',d['exact_definition_PPM_gt1_atleast5'],'screen',d['exact_screen_products'],'hit',d['exact_screen_hit_products'])

if __name__ == '__main__':
    main()
