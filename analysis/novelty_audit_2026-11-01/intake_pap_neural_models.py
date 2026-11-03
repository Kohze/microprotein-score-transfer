"""Pin the published PAP neural weights and training/inference definitions."""
from pathlib import Path
import hashlib
import json
import tarfile
import requests
import h5py

P=Path(__file__).resolve().parent
O=P/'pap_neural_comparison'
O.mkdir(exist_ok=True)
COMMIT='9b3eafa6bd9730c558cd018063fa0180edf9db1a'


def main():
    rows=[]
    for name in ['models/pap_weights.tgz','models/train_cnn2w.py','models/PAP.py']:
        url=f'https://raw.githubusercontent.com/KULL-Centre/_2025_Voutsinos_degron_cytosol/{COMMIT}/{name}'
        file=O/Path(name).name
        if not file.exists():
            response=requests.get(url,timeout=30)
            response.raise_for_status()
            file.write_bytes(response.content)
        rows.append(dict(file=file.name,url=url,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),bytes=file.stat().st_size))
    extracted=[]
    with tarfile.open(O/'pap_weights.tgz','r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            target=(O/member.name).resolve()
            assert target.is_relative_to(O.resolve()),member.name
            assert target.suffix=='.h5',member.name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(member).read())
            layers=[]
            with h5py.File(target,'r') as h:
                def inspect(name,obj):
                    if isinstance(obj,h5py.Dataset):
                        layers.append(dict(name=name,shape=list(obj.shape),dtype=str(obj.dtype)))
                h.visititems(inspect)
            extracted.append(dict(file=str(target.relative_to(O)),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),datasets=layers))
    receipt=dict(status='published numeric weights and source retrieved; no inference/evaluation yet',commit=COMMIT,
        downloads=rows,weights=extracted,serialization='HDF5 numeric weights only, no pickle')
    (O/'SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
