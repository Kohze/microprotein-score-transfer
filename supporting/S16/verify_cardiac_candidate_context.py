"""Verify links against original XLSX XML, independently of openpyxl extraction."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,posixpath,zipfile,xml.etree.ElementTree as E

P=Path(__file__).resolve().parent
D=P/'detection_evolution_prior'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def sheet_rows(path,name):
    with zipfile.ZipFile(path) as z:
        wb=E.fromstring(z.read('xl/workbook.xml'))
        sheet=next(s for s in wb.findall('s:sheets/s:sheet',NS) if s.get('name')==name)
        rid=sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        rels=E.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target=next(r.get('Target') for r in rels if r.get('Id')==rid)
        member=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in x.findall('.//s:t',NS)) for x in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        result=[]
        for row in E.fromstring(z.read(member)).findall('s:sheetData/s:row',NS):
            cells={}
            for c in row.findall('s:c',NS):
                letters=''.join(a for a in c.get('r','') if a.isalpha());col=0
                for a in letters:col=col*26+ord(a)-64
                value=c.find('s:v',NS);v=value.text if value is not None else None
                if c.get('t')=='s' and v is not None:v=strings[int(v)]
                elif c.get('t')=='inlineStr':v=''.join(t.text or '' for t in c.findall('.//s:t',NS))
                elif c.get('t') not in ['s','str','inlineStr'] and v is not None:
                    try:v=float(v)
                    except ValueError:pass
                cells[col-1]=v
            result.append((int(row.get('r')),cells))
        return result

def main():
    a=json.loads((D/'CARDIAC_CANDIDATE_AND_PERTURBATION_AUDIT.json').read_text())
    for v in a['sources'].values():assert sha(Path(v['path']))==v['sha256']
    with Path(a['sources']['candidate_roster']['path']).open(encoding='utf-8-sig',newline='') as f:roster=list(csv.DictReader(f))
    with (D/'CARDIAC_CANDIDATE_CONTEXT.csv').open(encoding='utf-8-sig',newline='') as f:output=list(csv.DictReader(f))
    assert len(roster)==len(output)==60
    original=sheet_rows(a['sources']['young_table']['path'],'Supplementary Table 6')
    records=[dict(orf=c.get(0),species=c.get(6),age=c.get(5),sequence=c.get(7),row=n) for n,c in original if n>=3 and c.get(0)]
    assert len(records)==504
    assert Counter(r['species'] for r in records)==Counter(a['young_table_species_counts'])
    assert Counter(r['age'] for r in records)==Counter(a['young_table_age_counts'])
    found=[]
    for c,o in zip(roster,output):
        assert c['candidate_key']==o['candidate_key'] and c['sequence']==o['sequence']
        hits=[r for r in records if r['sequence'].rstrip('*')==c['sequence']]
        human=[r for r in hits if r['species']=='human'];other=[r for r in hits if r['species']!='human']
        assert len(human)==int(o['exact_young_table_human_matches']) and len(other)==int(o['nonhuman_sequence_matches'])
        assert o['human_cardiac_ORFs']==';'.join(r['orf'] for r in human)
        assert o['human_age_labels']==';'.join(r['age'] for r in human)
        assert o['native_function_label_assigned']=='False'
        found += [(c['candidate_key'],r['orf'],r['species']) for r in hits]
    assert sorted(found)==sorted((r['candidate_key'],r['cardiac']['orf'],r['cardiac']['species']) for r in a['matched_details'])
    assert len([x for x in found if x[2]=='human'])==5 and len([x for x in found if x[2]!='human'])==1
    assert a['human_matches_with_primary_reporter_records']==2
    for target in ['LINC01405','SRP14-AS1']:
        values=[]
        for n,c in sheet_rows(a['sources']['source_data_Fig5']['path'],'CRISPRi '+target):
            # Gene names must be original text cells. Numeric/date-formatted
            # cells are not converted into apparent gene-name strings.
            if n==1 or not isinstance(c.get(0),str) or not c.get(0):continue
            try:p=float(c.get(1));fc=float(c.get(2))
            except (TypeError,ValueError):continue
            if math.isfinite(p) and math.isfinite(fc):values.append((c[0],p,fc))
        expected=a['CRISPRi_source_table_audit'][target]
        assert len(values)==expected['numeric_rows']==36611
        sig=[(g,p,fc) for g,p,fc in values if p<.05 and abs(fc)>.5]
        other=[r for r in sig if r[0]!=target]
        assert len(sig)==expected['significant_including_target']
        assert sum(fc>.5 for g,p,fc in other)==expected['downstream_up']
        assert sum(fc<-.5 for g,p,fc in other)==expected['downstream_down']
        assert sorted(g for g,p,fc in other)==sorted(r['gene'] for r in expected['significant_downstream_genes'])
    assert not a['new_native_function_label'] and not a['new_predictor_fit_or_inference']
    out=dict(status='passed',independent_reader='Raw XLSX ZIP/XML and CSV;no openpyxl dependency',source_young_ORF_rows=504,candidates_checked=60,human_exact_matches=5,nonhuman_sequence_matches=1,matching_primary_reporter_candidates=2,CRISPRi_numeric_rows_checked=73222,new_function_label_assigned=False,new_native_or_independent_reporter_validation=False)
    (D/'CARDIAC_CANDIDATE_AUDIT_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
