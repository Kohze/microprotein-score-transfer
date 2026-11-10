"""Independent pandas replay of the openpyxl/csv primary-table join."""
from pathlib import Path
import hashlib
import json
import math
import pandas as pd

P = Path(__file__).resolve().parent
M = P / 'ms_evidence_standards'


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main():
    audit = json.loads((M / 'CURRENT_CANDIDATE_MS_AUDIT.json').read_text(encoding='utf-8'))
    saved = json.loads((M / 'current_candidate_ms_evidence.json').read_text(encoding='utf-8'))
    roster = json.loads((P / 'native_system_prior/candidate_primaries/candidate_native_evidence.json').read_text(encoding='utf-8'))
    for r in audit['sources']:
        assert hashlib.sha256((P.parent / r['path']).read_bytes()).hexdigest() == r['sha256']
    data = {n: pd.read_excel(M / f'Deutsch_TableS{n}.xlsx') for n in [3, 6, 7, 9, 12]}
    tables = {n: d.loc[:, ~d.columns.astype(str).str.startswith('Unnamed:')]
              for n, d in data.items()}
    checked_cells = 0
    assert len(saved) == len(roster) == 60
    for c, r in zip(roster, saved):
        assert (c['candidate_key'], c['candidate_name'], c['sequence']) == (
            r['candidate_key'], r['candidate_name'], r['sequence'])
        exact = tables[9].loc[tables[9]['sequence'].eq(c['sequence'])]
        expected = clean(exact.to_dict('records'))
        assert r['exact_S9_rows'] == expected
        checked_cells += sum(len(x) for x in expected)
        ids = set(exact['orf_name'])
        for n in [3, 6, 7, 12]:
            expected = clean(tables[n].loc[tables[n]['Ribo-Seq_ORF'].isin(ids)].to_dict('records'))
            assert r['downstream_rows'][str(n)] == expected, (c['candidate_name'], n)
            checked_cells += sum(len(x) for x in expected)
    all_psms = pd.read_csv(M / 'Wacholder_41467_2025_68002_MOESM3_ESM.csv', low_memory=False)
    rated = pd.read_csv(M / 'Wacholder_41467_2025_68002_MOESM5_ESM.csv')
    psm = all_psms.loc[all_psms['orf_name'].eq('human-lv-merged.30212.4_238_442') &
                       all_psms['unmodified_peptide'].eq('CRGIQTLENVPPEFPASSK')]
    assert len(psm) == 1
    usi = psm.iloc[0]['usi']
    r = rated.loc[rated['USI'].eq(usi)]
    assert len(r) == 1 and r.iloc[0]['Rating..1.5.'] == 1 and r.iloc[0]['curator'] == 'F'
    assert audit['JHDM_exact_USI_ratings'][0]['USI'] == usi
    assert not all_psms['unmodified_peptide'].eq('ASMEPPLVAGGLSPR').any()
    assert not rated['peptide'].eq('ASMEPPLVAGGLSPR').any()
    shared = tables[9].loc[tables[9]['orf_name'].eq('c12riboseqorf86')]
    assert len(shared) == 1 and 'EVCLILKLK' in shared.iloc[0]['sequence']
    result = dict(status='passed', roster=60, exact_S9_candidates=45,
                  source_table_cells_checked=checked_cells,
                  JHDM_shotgun_exact_USI_curator_rating_verified=True,
                  RP11_HLA_signature_shared_with_other_ORF_verified=True,
                  raw_spectra_reprocessed=False, new_functional_or_tier_labels_assigned=False)
    (M / 'INDEPENDENT_CURRENT_MS_VERIFICATION.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
