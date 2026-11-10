"""Independent physical-OOXML verification; ignores declared sheet dimensions."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import posixpath
import re
import xml.etree.ElementTree as E
import zipfile

P=Path(__file__).resolve().parent
D=P/'native_detection_prior'
NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def read_xlsx(path,names,header=3):
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=E.fromstring(z.read('xl/sharedStrings.xml'))
            strings=[''.join(si.itertext()) for si in root]
        rels={r.attrib['Id']:r.attrib['Target'] for r in E.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets={s.attrib['name']:rels[s.attrib['{'+NS['r']+'}id']]
                for s in E.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS)}
        result={}
        for name in names:
            target=sheets[name]
            target=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            root=E.fromstring(z.read(target));grid=[]
            for row in root.findall('s:sheetData/s:row',NS):
                rownum=int(row.attrib['r']);cells={}
                for c in row.findall('s:c',NS):
                    letters=re.match('[A-Z]+',c.attrib['r']).group(0)
                    col=0
                    for x in letters:col=col*26+ord(x)-64
                    kind=c.get('t');v=c.find('s:v',NS)
                    if kind=='inlineStr':value=''.join(c.find('s:is',NS).itertext())
                    elif v is None:value=None
                    elif kind=='s':value=strings[int(v.text)]
                    elif kind=='b':value=v.text=='1'
                    elif kind in ['str','e']:value=v.text
                    else:value=float(v.text) if v.text is not None else None
                    cells[col-1]=value
                grid.append((rownum,cells))
            cols=next(c for n,c in grid if n==header)
            result[name]=[{key:c.get(i) for i,key in cols.items() if key is not None}
                          for n,c in grid if n>header and c.get(0) is not None]
    return result


def main():
    audit=json.loads((D/'BRAIN_NATIVE_CONTEXT_AUDIT.json').read_text(encoding='utf-8'))
    source=P.parent/audit['source']['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['source']['sha256']
    saved=json.loads((D/'brain_candidate_sequence_evidence.json').read_text(encoding='utf-8'))
    specs={'S1 Discovery':'Microprotein Sequence','S2 Proteomics':'sequence',
           'S3 Tryptic Peptides':'sequence','S5 ShortStop':'Microprotein Sequence'}
    data=read_xlsx(source,specs)
    assert {k:len(v) for k,v in data.items()}==audit['table_rows']
    cells=0
    for candidate in saved:
        for name,col in specs.items():
            expected=[r for r in data[name] if r[col]==candidate['sequence']]
            assert expected==candidate['source_rows'][name],(candidate['candidate_key'],name)
            cells+=sum(len(r) for r in expected)
    positive=[r for r in data['S2 Proteomics'] if r['Status']=='Unreviewed' and r['total_unique_spectral_counts']>0]
    assert len(positive)==audit['unreviewed_with_positive_unique_PSM_counts']==3217
    assert sum(31<=len(r['sequence'])<=150 and set(r['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') for r in positive)==2353
    s3seq={r['sequence'] for r in data['S3 Tryptic Peptides']}
    assert sum(r['sequence'] in s3seq for r in positive)==3195
    # Source Figure7 endpoints explicitly contain one row per well, not a
    # row-level experiment/batch identifier. Do not invent biological replicate IDs.
    fig7=read_xlsx(D/'BrainAtlas2026/43587_2026_1207_MOESM11_ESM.xlsx',
                   ['H_Basal_OCR','I_Max_Capacity','J_ATP_Linked_OCR','K_Reserve_OCR'],header=1)
    counts={k:dict(Counter(r['group'] for r in v)) for k,v in fig7.items()}
    assert all(sum(c.values())==71 for c in counts.values())
    assert all(sorted(c.values())==[23,24,24] for k,c in counts.items() if k!='J_ATP_Linked_OCR')
    assert counts['J_ATP_Linked_OCR']=={'NTC':23,'Micro-MKKS63(1)':21,'Micro-MKKS63(2)':27}
    fvalues={}
    for name,rows in fig7.items():
        key=next(k for k in rows[0] if k!='group')
        groups=[[r[key] for r in rows if r['group']==group] for group in counts[name]]
        grand=sum(sum(g) for g in groups)/71
        means=[sum(g)/len(g) for g in groups]
        between=sum(len(g)*(mean-grand)**2 for g,mean in zip(groups,means))
        within=sum(sum((x-mean)**2 for x in g) for g,mean in zip(groups,means))
        fvalues[name]=(between/2)/(within/68)
    assert [round(v,2) for v in fvalues.values()]==[8.56,8.96,14.12,8.84]
    result=dict(status='passed',roster=60,source_candidate_cells_verified=cells,
        full_source_table_rows_verified=audit['table_rows'],independent_reader='stdlib zipfile + physical worksheet XML',
        declared_sheet_dimensions_ignored=True,PSM_count_endpoint_feasibility_verified=True,
        source_Fig7_per_well_counts=counts,experiment_ID_column_in_endpoints=False,
        source_Fig7_rounded_ANOVA_F_reproduced=fvalues,
        unresolved_source_reporting='Fig7 caption gives23/24/24 for all endpoints; ATP-linked workbook labels give23/21/27. Physical XML and openpyxl agree; labels retained without correction.',
        new_predictor_associations_or_function_labels=False)
    (D/'INDEPENDENT_BRAIN_NATIVE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
