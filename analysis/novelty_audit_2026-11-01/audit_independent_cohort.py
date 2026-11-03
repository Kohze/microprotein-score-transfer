"""Resolve author holdout membership without inspecting assay score values."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import requests

P = Path(__file__).resolve().parent
D = P / 'independent_data'


def main():
    tree = json.loads((P / 'sources/Voutsinos2026_repository_tree.json').read_text())
    commit = tree['sha']
    file = D / 'Voutsinos_cytosol_tiles.csv.gz'
    url = f'https://raw.githubusercontent.com/KULL-Centre/_2025_Voutsinos_degron_cytosol/{commit}/library/cytosol_tiles.csv.gz'
    if not file.exists():
        r = requests.get(url, timeout=45)
        r.raise_for_status()
        file.write_bytes(r.content)
    header = pd.read_csv(file, nrows=0).columns.tolist()
    print('Library columns', header, flush=True)
    lib = pd.read_csv(file, usecols=['name', 'set'])
    assert not lib.name.duplicated().any()
    # The assay score columns are deliberately not loaded at this stage.
    allrows = pd.read_csv(D / 'Voutsinos_degron_scores.csv.gz', usecols=['name', 'aa'])
    train = pd.read_csv(D / 'Voutsinos_degron_train.csv.gz', usecols=['name', 'aa', 'set'])
    joined = allrows.merge(lib, on='name', validate='one_to_one', how='left')
    assert joined['set'].notna().all()
    test = joined[joined['set'].eq('test')].copy()
    trainseq = set(train.aa)
    test['exact_peptide_in_author_training'] = test.aa.isin(trainseq)
    test['protein_id'] = test.name.str.split('_').str[0]
    test['length'] = test.aa.str.len()
    test['valid_standard_30mer'] = test.aa.map(lambda seq: len(seq) == 30 and set(seq) <= set('ACDEFGHIKLMNPQRSTVWY'))
    guo = pd.read_parquet(P.parent / 'stability_benchmark_2026-10-31/natural_assay_records.parquet')
    test['exact_Guo_natural_peptide'] = test.aa.isin(set(guo.protein_sequence))
    test.to_parquet(D / 'Voutsinos_author_holdout_sequence_audit.parquet', index=False)
    test.to_csv(D / 'Voutsinos_author_holdout_sequence_audit.csv', index=False)
    result = dict(status='outcome-free membership audit complete; related-sequence and gene audit still required',
        pinned_commit=commit, library_url=url, library_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        library_rows=len(lib), scored_rows=len(joined), author_train_rows=len(train),
        author_train_unique_peptides=int(train.aa.nunique()), author_split_counts=joined['set'].value_counts().to_dict(),
        author_test_rows=len(test), author_test_unique_peptides=int(test.aa.nunique()),
        author_test_source_gene_ids=int(test.protein_id.nunique()),
        exact_test_peptides_in_author_training=int(test.exact_peptide_in_author_training.sum()),
        measured_standard_30mer_test_rows=int(test.valid_standard_30mer.sum()),
        exact_Guo_natural_matches=int(test.exact_Guo_natural_peptide.sum()),
        assay_outcome_columns_loaded=False, model_predictions_evaluated=False,
        next_required_checks=['map protein/gene identities and exclude source-shared genes',
            'audit operational peptide similarity against all Guo and annotation inputs',
            'confirm author PAP training/holdout split and source model feature provenance',
            'freeze prediction and reporting rules before inspecting new transfer outcomes'])
    (P / 'INDEPENDENT_COHORT_FEASIBILITY.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
