"""Recreate only the current Fig3 using its saved source tables."""
from pathlib import Path
import importlib.util,json,hashlib

P=Path(__file__).resolve().parent
source=P/'make_publication_figures.py'
if not source.exists():source=P.parent/'analysis/make_publication_figures.py'
spec=importlib.util.spec_from_file_location('publication_plots',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.P=P;m.S=P/'figures/source_data';m.F=P/'figures/main';m.setup()
original_save=m.save
def save_current(fig,name):
    assert len(fig.axes[1].lines)==4
    assert all(line.get_linestyle()=='None' and len(line.get_xdata())==1 for line in fig.axes[1].lines)
    original_save(fig,'Fig3')
m.save=save_current;m.figure6()
mapping=json.loads((P/'figures/PUBLICATION_NUMBERING_MAP.json').read_text())
mapping['main/Fig3']['pdf_sha256']=hashlib.sha256((P/'figures/main/Fig3.pdf').read_bytes()).hexdigest()
(P/'figures/PUBLICATION_NUMBERING_MAP.json').write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8')
print('Updated Fig3 PDF, SVG and PNG from saved values. Other figures were not rebuilt.')
