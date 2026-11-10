"""Read original XLSX physical values without interpreting or repairing styles.

Some deposited workbooks have invalid style references or padded dimensions.
Numeric/date-formatted cells remain numeric; no gene or peptide is inferred.
"""
from pathlib import Path
import posixpath,zipfile,xml.etree.ElementTree as E

NS={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def sheets(path):
    with zipfile.ZipFile(path) as z:
        w=E.fromstring(z.read('xl/workbook.xml'));rels=E.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        result=[]
        for sh in w.findall('s:sheets/s:sheet',NS):
            rid=sh.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            target=next(r.get('Target') for r in rels if r.get('Id')==rid)
            member=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl',target))
            result.append((sh.get('name'),member))
        return result


def rows(path,name,columns=None):
    with zipfile.ZipFile(path) as z:
        member=dict(sheets(path))[name];strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(t.text or '' for t in x.findall('.//s:t',NS)) for x in E.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        with z.open(member) as f:
            for _,row in E.iterparse(f,events=['end']):
                if row.tag!='{'+NS['s']+'}row':continue
                d={}
                for c in row.findall('s:c',NS):
                    col=0
                    for a in ''.join(x for x in c.get('r','') if x.isalpha()):col=col*26+ord(a)-64
                    col-=1
                    if columns is not None and col not in columns:continue
                    node=c.find('s:v',NS);value=node.text if node is not None else None;kind=c.get('t')
                    if kind=='s' and value is not None:value=strings[int(value)]
                    elif kind=='inlineStr':value=''.join(t.text or '' for t in c.findall('.//s:t',NS))
                    elif kind=='b' and value is not None:value=bool(int(value))
                    elif kind not in ['str','e'] and value is not None:
                        try:value=float(value)
                        except ValueError:pass
                    if value is not None:d[col]=value
                n=int(row.get('r'));row.clear()
                if d:yield n,d
