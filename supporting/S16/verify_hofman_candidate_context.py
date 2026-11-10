"""Recheck exact Hofman joins using original XLSX ZIP/XML, without openpyxl."""
from pathlib import Path
from collections import Counter
import csv,json
from verify_cardiac_candidate_context import sheet_rows,sha

P=Path(__file__).resolve().parent
S=P/'detection_evolution_prior/Hofman2024'
W=S/'repository_supplement'

def clean(s):
    return s[:-1] if isinstance(s,str) and s.endswith('*') else s

def main():
    a=json.loads((S/'CANDIDATE_ASSAY_AUDIT.json').read_text())
    for name,h in a['source_paths'].items():assert sha(Path(name))==h,name
    roster=P/'native_function_bridge/candidate_bridge.csv'
    assert sha(roster)==a['roster_sha256']
    with roster.open(encoding='utf-8-sig',newline='') as f: candidates=list(csv.DictReader(f))
    with (S/'CANDIDATE_ASSAY_CONTEXT.csv').open(encoding='utf-8-sig',newline='') as f: outputs=list(csv.DictReader(f))
    defs=[(n,c) for n,c in sheet_rows(next(W.glob('*mmc2.xlsx')),'Table S1P') if n>=4 and c.get(0)]
    samples={c[0]:c.get(1) for n,c in sheet_rows(next(W.glob('*mmc2.xlsx')),'Table S1N') if n>=4 and c.get(0)}
    library=[(n,c) for n,c in sheet_rows(next(W.glob('*mmc3.xlsx')),'Table_S2A') if n>=4 and isinstance(c.get(5),str) and c.get(5)]
    screen={c[0]:(n,c) for n,c in sheet_rows(next(W.glob('*mmc3.xlsx')),'Table_S2E') if n>=4 and c.get(0)}
    assert len(defs)==8008 and len(candidates)==len(outputs)==60
    totals=Counter();positives=[]
    for c,o in zip(candidates,outputs):
        assert c['candidate_key']==o['candidate_key'] and c['sequence']==o['sequence']
        ds=[(n,d) for n,d in defs if clean(d.get(11))==c['sequence']]
        assert len(ds)==int(o['exact_definition_records'])
        n5=sum(samples[d[0]]>=5 for n,d in ds)
        assert n5==int(o['exact_definition_PPM_gt1_atleast5'])
        saved=json.loads(o['source_definitions'])
        assert [(n,d[0],samples[d[0]]) for n,d in ds]==[(d['source_excel_row'],d['orf_id'],d['samples_PPM_gt1']) for d in saved]
        ls={}
        for n,l in library:
            if clean(l[5])==c['sequence']:ls.setdefault((l[0],l[1]),[]).append(n)
        assert len(ls)==int(o['exact_screen_products'])
        hits=0;resolved=0;unknown=0
        sl=json.loads(o['source_screen_products'])
        assert len(sl)==len(ls)
        for d in sl:
            assert d['rows']==ls[(d['name'],d['type'])]
            name=d['name']+'_'+{'uoORF':'overlapping.uORF'}.get(d['type'],d['type'])
            assert name==d['summary_annotation']
            if name not in screen:
                assert not d['summary_found'] and d['source_hit'] is None
                unknown+=1
                continue
            n,e=screen[name];resolved+=1
            assert d['source_summary_excel_row']==n and d['source_hit']==e.get(5)
            hits+=e.get(5)==1
            if e.get(5)==1:positives.append(c['candidate_key'])
        assert hits==int(o['exact_screen_hit_products']) and resolved==int(o['exact_screen_resolved_summaries']) and unknown==int(o['exact_screen_unresolved_summaries'])
        assert o['native_peptide_function_validated_by_this_join']=='False'
        totals.update(dict(definition_candidates=bool(ds),definition_records=len(ds),translated_candidates=n5>0,screen_candidates=bool(ls),resolved_candidates=resolved>0,unknown_candidates=unknown>0,hit_candidates=hits>0))
    assert dict(totals)==dict(definition_candidates=49,definition_records=49,translated_candidates=33,screen_candidates=21,resolved_candidates=20,unknown_candidates=1,hit_candidates=4)
    category=None;counts=Counter()
    for n,d in sheet_rows(next(W.glob('*mmc3.xlsx')),'Table_S2L'):
        if n<5:continue
        if d.get(0):category=d[0]
        if isinstance(d.get(1),str):counts[category]+=1
    assert dict(counts)==a['tiling_classification_counts'] and sum(counts.values())==50
    asdurf=[d[5] for n,d in library if d.get(0)=='ASNSD1' and d.get(1)=='uORF']
    assert len(asdurf)==4 and set(asdurf)=={a['ASNSD1_96aa_sequence']}
    assert all(len(s)==96 and s not in {c['sequence'] for c in candidates} for s in asdurf)
    assert a['ASNSD1_exact_roster_matches']==0
    assert not a['new_native_or_edited_function_measurement']
    out=dict(status='passed',independent_reader='Original XLSX ZIP/XML; no openpyxl',candidates=60,source_definitions=8008,source_definition_exact_candidates=49,PPM_gt1_atleast5_exact_candidates=33,exact_screen_candidates=21,screen_summary_resolved_candidates=20,screen_summary_unresolved_candidates=1,source_screen_hit_candidates=4,source_screen_hit_keys=positives,tiling_records=50,ASDURF96_roster_exact_matches=0,new_native_or_edited_function_measurement=False)
    (S/'CANDIDATE_ASSAY_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
