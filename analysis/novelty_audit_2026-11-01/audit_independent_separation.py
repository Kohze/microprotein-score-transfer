"""Operational source-gene and peptide separation, without target outcomes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
B = P.parent / 'stability_benchmark_2026-10-31'
sys.path.insert(0, str(B / 'vendor'))
import parasail


def main():
    test = pd.read_parquet(P / 'independent_data/Voutsinos_author_holdout_sequence_audit.parquet')
    groups = pd.read_parquet(B / 'sequence_dependence_groups.parquet')
    source = groups[groups.sources.str.contains('Guo|annotation_reference', regex=True)].copy()
    genes = {g.split('.')[0] for v in source.genes.fillna('') for g in v.split(';') if g}
    # The library names start with ENSG identifiers; they are gene identifiers,
    # not ENSP protein accessions. We retain that distinction explicitly.
    test['source_gene_overlap'] = test.protein_id.isin(genes)
    sources = source.sequence.tolist()
    aa = 'ACDEFGHIKLMNPQRSTVWY'
    lengths = np.array(list(map(len, sources)))
    counts = np.array([[s.count(c) for c in aa] for s in sources], dtype=np.int16)
    started = time.time()
    def scan(row):
        seq = row.aa
        count = np.array([seq.count(c) for c in aa])
        candidates = np.flatnonzero((lengths >= .8 * len(seq)) & (.8 * lengths <= len(seq)))
        bound = np.minimum(count, counts[candidates]).sum(1)
        candidates = candidates[bound + 1e-9 >= .32 * np.maximum(lengths[candidates], len(seq))]
        hits = []
        for j in candidates:
            alignment = parasail.sw_trace_striped_16(seq, sources[j], 10, 1, parasail.blosum62)
            q, r = alignment.traceback.query, alignment.traceback.ref
            if not q:
                continue
            ident = sum(x == y and x != '-' for x, y in zip(q, r)) / len(q)
            if ident < .4:
                continue
            covq, covr = sum(x != '-' for x in q) / len(seq), sum(x != '-' for x in r) / len(sources[j])
            if covq >= .8 and covr >= .8:
                hits.append(dict(target_name=row.name, source_sequence=sources[j],
                    source_groups=source.iloc[j].sources, identity=ident, coverage_target=covq, coverage_source=covr))
        return hits, len(candidates)
    edges, aligned, done = [], 0, 0
    # Audit every author-held-out sequence, including gene-linked rows, so the
    # separate exclusion reasons can be inspected without outcome conditioning.
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(scan, row) for row in test.itertuples()]
        for task in as_completed(tasks):
            hits, n = task.result()
            edges.extend(hits)
            aligned += n
            done += 1
            if done % 1000 == 0:
                progress = dict(complete=done, total=len(test), aligned_pairs=aligned, edges=len(edges),
                    elapsed_seconds=round(time.time()-started, 1))
                (P / 'SEPARATION_PROGRESS.json').write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
    edge = pd.DataFrame(edges, columns=['target_name', 'source_sequence', 'source_groups', 'identity', 'coverage_target', 'coverage_source'])
    edge.to_parquet(P / 'independent_data/target_source_similarity_edges.parquet', index=False)
    for label, cut in [('primary60', .6), ('sensitivity40', .4)]:
        blocked = set(edge.loc[edge.identity.ge(cut), 'target_name'])
        test[label + '_similarity_overlap'] = test.name.isin(blocked)
        test[label + '_eligible'] = (test.valid_standard_30mer & ~test.exact_peptide_in_author_training &
                                      ~test.source_gene_overlap & ~test.name.isin(blocked))
    test.to_parquet(P / 'independent_data/separated_author_holdout.parquet', index=False)
    receipt = dict(status='outcome-free operational separation complete; author release-count discrepancy still open',
        author_holdout_rows=len(test), source_native_reference_sequences=len(source),
        source_genes=len(genes), source_gene_linked_rows=int(test.source_gene_overlap.sum()),
        primary60_eligible=int(test.primary60_eligible.sum()), sensitivity40_eligible=int(test.sensitivity40_eligible.sum()),
        primary60_gene_groups=int(test.loc[test.primary60_eligible, 'protein_id'].nunique()),
        sensitivity40_gene_groups=int(test.loc[test.sensitivity40_eligible, 'protein_id'].nunique()),
        eligible_pairs_aligned=aligned, similarity_edges=len(edge), target_outcomes_used=False,
        algorithm='Smith-Waterman BLOSUM62; open 10, extend 1; identity over alignment columns; >=80% coverage of both sequences',
        source_manifest_sha256=hashlib.sha256((B / 'sequence_dependence_groups.parquet').read_bytes()).hexdigest(),
        limitations=['operational separation does not exclude remote homology',
                    'PAP feature training independence relies additionally on the original author holdout audit',
                    '15,818 public test rows versus 15,815 described in the final paper requires resolution'])
    (P / 'INDEPENDENT_SEPARATION_AUDIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
