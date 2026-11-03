"""Freeze pre-existing composition-model predictions, without outcome fitting."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.special import expit

P = Path(__file__).resolve().parent
O = P/'published_predictor_comparison'
B = P.parent/'stability_benchmark_2026-10-31'
AA = 'ACDEFGHIKLMNPQRSTVWY'


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def literal_model(file,name):
    out = {}
    for statement in ast.parse(file.read_text(encoding='utf-8')).body:
        if not isinstance(statement,ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target,ast.Name) and target.id == name:
            out = ast.literal_eval(statement.value)
        elif isinstance(target,ast.Subscript):
            keys = []
            while isinstance(target,ast.Subscript):
                keys.insert(0,ast.literal_eval(target.slice))
                target = target.value
            if not isinstance(target,ast.Name) or target.id != name:
                continue
            value = ast.literal_eval(statement.value)
            node = out
            for key in keys[:-1]:
                node = node.setdefault(key,{})
            node[keys[-1]] = value
    return out


def main():
    original = O/'QCDpred.py'
    papfile = B/'inputs/Voutsinos2026_code/PAP.py'
    qcd = literal_model(original,'model')
    yeast = literal_model(papfile,'yeast17')
    human = literal_model(papfile,'human25')
    assert set(qcd) == set(AA)|{'intersect'}
    assert qcd['intersect'] == yeast['intercept']
    assert all(qcd[a] == yeast['comp'][a] for a in AA)
    # All target and Guo sequence inputs at this step exclude measured outcomes.
    target = pd.read_parquet(P/'independent_data/Voutsinos_author_holdout_sequence_audit.parquet',columns=['aa'])
    source = pd.read_parquet(B/'natural_assay_records.parquet',columns=['protein_sequence'])
    seqs = sorted(set(target.aa)|set(source.protein_sequence))
    training = pd.read_csv(O/'Mashahreh_wt_tiles.csv.gz',sep=';')
    print('Training columns',training.columns.tolist(),flush=True)
    standard = training.peptide.map(lambda s:len(s)==17 and set(s)<=set(AA))
    valid = standard & training['wt.raw_sum'].gt(50) & (training['wt.psi'].lt(2.2)|training['wt.psi'].gt(2.8))
    trainseq = set(training.loc[valid,'peptide'])
    rows = []
    for seq in seqs:
        windows = [seq[i:i+17] for i in range(len(seq)-16)] if len(seq)>=17 else []
        prob = [expit(qcd['intersect']+sum(qcd[a] for a in window)) for window in windows]
        human_first = seq[:25]
        rows.append(dict(sequence=seq,length=len(seq),n_complete_17_windows=len(windows),
            QCDPred_mean17_stability=1-float(np.mean(prob)) if windows else np.nan,
            QCDPred_normalized_cargo_stability=1-float(expit(qcd['intersect']+sum(qcd[a] for a in seq)*17/len(seq))),
            human25_first25_stability=1-float(expit(human['intercept']+sum(human['comp'][a] for a in human_first)*25/len(human_first))),
            human25_normalized_cargo_stability=1-float(expit(human['intercept']+sum(human['comp'][a] for a in seq)*25/len(seq))),
            exact_labeled_QCDPred_training_window=any(w in trainseq for w in windows)))
    predictions = pd.DataFrame(rows)
    # Verify original CLI on deterministic sequences; no target association involved.
    samples = ['CALLQSRLLLSAPRRAAATARY'] + [s for s in seqs if len(s)>=17][::max(1,len(seqs)//8)][:8]
    cli = subprocess.run([sys.executable,str(original),*samples],capture_output=True,text=True,check=True)
    observed = [float(line.split()[2]) for line in cli.stdout.splitlines() if line.strip()]
    expected = [float(expit(qcd['intersect']+sum(qcd[a] for a in s[i:i+17])))
                for s in samples for i in range(len(s)-16)]
    assert len(observed) == len(expected)
    error = float(np.max(abs(np.array(observed)-np.array(expected))))
    assert error <= 5.001e-6  # the official CLI prints five decimals
    file = O/'frozen_published_predictions.parquet'
    predictions.to_parquet(file,index=False)
    predictions.to_csv(file.with_suffix('.csv'),index=False)
    receipt = dict(status='frozen published coefficients evaluated; no target outcome fitting',
        protocol_sha256=sha(O/'COMPARISON_PROTOCOL.txt'),source_code_sha256=sha(Path(__file__)),
        QCDPred_source_sha256=sha(original),PAP_source_sha256=sha(papfile),prediction_sha256=sha(file),
        source_sequences=len(predictions),QCDPred_labeled_training_rows=int(valid.sum()),
        training_row_audit='14 nonstandard stop-containing rows excluded, reproducing the published 13,559 labeled peptides',
        QCDPred_unique_training_peptides=len(trainseq),official_CLI_checked_windows=len(observed),
        CLI_max_rounding_error=error,yeast17_coefficients_identical_to_original=True,
        QCDPred_higher_score_definition='1 minus mean degron probability across complete consecutive 17mers',
        human25_provenance='target-study-trained comparator, not an external training source',
        design='secondary baseline challenge after original target A/T results were already seen',
        target_outcomes_read=False,target_outcomes_used_for_fitting=False)
    (O/'BASELINE_PREDICTION_FREEZE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
