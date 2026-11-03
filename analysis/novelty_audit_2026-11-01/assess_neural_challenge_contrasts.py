"""Secondary uncertainty for the observed strong-baseline/window challenge."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from evaluate_pap_neural_models import bootstrap

P=Path(__file__).resolve().parent
O=P/'pap_neural_comparison'

def main():
    (O/'SECONDARY_CONTRAST_PROTOCOL.txt').write_text('''SECONDARY CONTRAST ADDENDUM
The neural model associations and original calibration/order intervals have
already been inspected. These additional paired contrasts quantify the observed
strong-baseline and input-window differences; they are descriptive, post hoc,
not a new preregistered discovery. Reuse the primary1743 common cohort, existing
1591 operational groups and1000 conditional resamples. No predictions, models,
cohorts or source calibration maps are changed. Include both CNN architectures
and raw/calibrated window contrasts, plus their direct full-cargo contrast.
''',encoding='utf-8')
    pairs=pd.read_parquet(P/'independent_interventions/evaluated_predictions.parquet')
    d=pd.read_parquet(O/'paired_delta_predictions.parquet')
    assert pairs.source_excel_row.equals(d.source_excel_row)
    mask=pairs.primary60_common_eligible
    c=[]
    for arch in ['cnn2w1','cnn2w7']:
        for suffix in ['', '__source_WT_calibrated']:
            c.append((arch+'__full_cargo__full'+suffix,arch+'__C30__full'+suffix,'full58 minus C30; post hoc'))
    c.append(('cnn2w7__full_cargo__full','cnn2w1__full_cargo__full','separately fitted CNN7 minus CNN1; not causal order'))
    # The original paired-score file uses d__ prefixes; derive directly from
    # its frozen endpoint scores, rather than assuming column names.
    pred=pd.read_parquet(P/'independent_interventions/frozen_predictions.parquet')
    key='PAP_full_cargo__source_WT_calibrated'
    if 'sequence' in pred:
        pred=pred.set_index('sequence')
    d[key]=pred.loc[pairs.variant_sequence,key].to_numpy()-pred.loc[pairs.wt_sequence,key].to_numpy()
    c.append(('cnn2w1__full_cargo__full',key,'published CNN1 raw minus source-calibrated linear PAP; post hoc'))
    values={m:d.loc[mask,m].to_numpy() for row in c for m in row[:2]}
    result=bootstrap(pairs.loc[mask,'observed_delta'].to_numpy(),pairs.loc[mask,'target_dependence_group'].to_numpy(),values,c,20261107)
    pd.DataFrame(result).to_csv(O/'secondary_challenge_contrasts.csv',index=False)
    print(pd.DataFrame(result).to_string(index=False))

if __name__=='__main__':
    main()
