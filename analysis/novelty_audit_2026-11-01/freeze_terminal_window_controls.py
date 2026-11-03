"""Freeze same-input23mer PAP controls after the first terminal results."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'
sys.path.insert(0,str(P.parent/'stability_benchmark_2026-10-31'))
import run_benchmark as b
import hashlib
import json
import numpy as np
import pandas as pd
from pap_neural_runtime import PublishedPAP

def main():
    protocol=O/'WINDOW_CONTEXT_ADDENDUM.txt'
    protocol.write_text('''SECONDARY MATCHED-INPUT WINDOW CHECK
DEGRONOPEDIA C23 and PAP full58 results have now been inspected. This addendum
is post hoc, intended to prevent interpreting the different input coverage as
an architecture effect. Existing frozen C30 CNN1/CNN7 predictions will be added
on the same four exposure strata. Additionally freeze both published neural
models on the identical C23 inputs consumed by DEGRONOPEDIA. No target fitting,
calibration, direction selection, weights or cohorts change. C23 and full58
are supported variable-length inference outside the training30mer length.

Use the same3302 input C23s, both released architectures and all four declared
strata. Verify every C23 full/internal/terminal score by independent NumPy
inference. Freeze before new C23-model associations. Report all C23/C30/full58
raw neural results; do not select a favorable window. Add paired conditional
contrasts of C-model minus CNN1 at C23 and C30 on all1743, endpoint-unmatched289
and available-symbol97 cohorts. Original comparisons/negative results remain.
These contrasts cannot isolate architecture because training sources also
differ, and the97 sensitivity is not certified gene independence.
''',encoding='utf-8')
    seq=pd.read_parquet(O/'frozen_predictions.parquet').C23_sequence.tolist()
    out=pd.DataFrame(dict(C23_sequence=seq));errors=[]
    for arch in ['cnn2w1','cnn2w7']:
        model=PublishedPAP(arch);values=model.predict(seq)
        for start in range(0,len(seq),512):
            expected=model.numpy_predict(seq[start:start+512])
            errors.append(float(np.max(abs(expected-values[start:start+512]))))
        out[arch+'__C23__full']=values[:,0]
    assert max(errors)<1e-5
    path=O/'frozen_window_controls.parquet';out.to_parquet(path,index=False)
    receipt=dict(status='C23 neural controls frozen before their new associations',
        prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),post_hoc_after_terminal_results=True,
        independent_numpy_scores_checked=len(seq)*2,numerical_max_error=max(errors),
        original_terminal_predictions_unchanged=True,target_fitting=False,calibration=False)
    (O/'WINDOW_CONTROL_FREEZE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':
    with b.threadpool_limits(limits=4):main()
