"""Read-only primary-table audit; exact full-sequence joins precede ORF-ID lookups.

This does not reassess spectra, assign new evidence tiers or infer functionality.
"""
from pathlib import Path
import csv
import hashlib
import json
import openpyxl

P = Path(__file__).resolve().parent
M = P / 'ms_evidence_standards'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(name, sheet=0):
    w = openpyxl.load_workbook(M / name, read_only=True, data_only=True)
    s = w.worksheets[sheet]
    rows = iter(s.values)
    headers = next(rows)
    result = [dict((h, row[i] if i < len(row) else None) for i, h in enumerate(headers))
              for row in rows if any(x is not None for x in row)]
    for r in result:
        r.pop(None, None)
    w.close()
    return result


def main():
    roster_path = P / 'native_system_prior/candidate_primaries/candidate_native_evidence.json'
    roster = json.loads(roster_path.read_text(encoding='utf-8'))
    tables = {n: records(f'Deutsch_TableS{n}.xlsx') for n in [3, 6, 7, 9, 12]}
    assert len(tables[9]) == 7264 and len(tables[6]) == 3116
    assert sum(r['group(d=1, u=0)'] for r in tables[9]) == 1785
    output = []
    for c in roster:
        exact = [r for r in tables[9] if r['sequence'] == c['sequence']]
        ids = {r['orf_name'] for r in exact}
        output.append(dict(candidate_key=c['candidate_key'], candidate_name=c['candidate_name'],
                           sequence=c['sequence'], exact_S9_rows=exact,
                           downstream_rows={str(n): [r for r in tables[n] if r['Ribo-Seq_ORF'] in ids]
                                            for n in [3, 6, 7, 12]}))
    jhdm = next(r for r in output if r['candidate_name'] == 'JHDM1D-AS1')
    rp11 = next(r for r in output if r['candidate_name'] == 'RP11-620J15.3')
    slco = next(r for r in output if r['candidate_name'] == 'SLCO5A1')
    assert [r['orf_name'] for r in jhdm['exact_S9_rows']] == ['c7riboseqorf137']
    assert [r['orf_name'] for r in rp11['exact_S9_rows']] == ['c12riboseqorf85']
    assert [r['orf_name'] for r in slco['exact_S9_rows']] == ['c8riboseqorf60']
    # Original ORF, unmodified peptide and USI all agree; no gene-name assignment.
    with (M / 'Wacholder_41467_2025_68002_MOESM3_ESM.csv').open(newline='', encoding='utf-8') as f:
        all_psms = list(csv.DictReader(f))
    with (M / 'Wacholder_41467_2025_68002_MOESM5_ESM.csv').open(newline='', encoding='utf-8') as f:
        ratings = list(csv.DictReader(f))
    original_id = 'human-lv-merged.30212.4_238_442'
    shotgun = 'CRGIQTLENVPPEFPASSK'
    target = 'ASMEPPLVAGGLSPR'
    exact_psm = [r for r in all_psms if r['orf_name'] == original_id and r['unmodified_peptide'] == shotgun]
    assert len(exact_psm) == 1 and shotgun in jhdm['sequence']
    usi = exact_psm[0]['usi']
    exact_ratings = [r for r in ratings if r['USI'] == usi]
    assert len(exact_ratings) == 1 and float(exact_ratings[0]['Rating..1.5.']) == 1
    assert not any(r['unmodified_peptide'] == target for r in all_psms)
    assert not any(r['peptide'] == target for r in ratings)
    shared_ids = rp11['downstream_rows']['6'][0]['Other.mappings'].split(',')
    shared_s9 = [r for r in tables[9] if 'CONTRIB_GENCODE_' + r['orf_name'] in shared_ids]
    peptide = rp11['downstream_rows']['6'][0]['sequence']
    assert peptide == 'EVCLILKLK' and peptide in rp11['sequence']
    assert len(shared_s9) == 1 and peptide in shared_s9[0]['sequence']
    assert slco['downstream_rows']['6'][0]['sequence'] == 'TPAPCGTRL'
    assert 'TPAPCGTRL' in slco['sequence']
    files = [roster_path] + [M / f'Deutsch_TableS{n}.xlsx' for n in tables] + [
        M / 'Wacholder_41467_2025_68002_MOESM3_ESM.csv',
        M / 'Wacholder_41467_2025_68002_MOESM5_ESM.csv']
    audit = dict(
        scope='Exploratory selected-panel source-table join; no new raw-spectrum analysis or function labels',
        normalization='None: full amino-acid sequence equality; ORF IDs used only after sequence equality',
        roster=len(roster), exact_S9_candidates=sum(bool(r['exact_S9_rows']) for r in output),
        S9_rows=len(tables[9]), S9_HLA_detected_rows=1785,
        JHDM_shotgun_PSM=exact_psm, JHDM_exact_USI_ratings=exact_ratings,
        JHDM_SRM_peptide=target,
        JHDM_SRM_in_Wacholder_tables=False,
        JHDM_SRM_scope='Main text explicitly excludes SRM from reassessment; missing rating is not a quality verdict',
        RP11_shared_peptide=peptide, RP11_other_ORF_rows=shared_s9,
        tier_scope='Source-reported final tiers in retrieved S7; single-HLA entries not all manually validated',
        missingness='No exact match or no table row is not evidence of absent translation, protein or function',
        sources=[dict(path=str(p.relative_to(P.parent)), sha256=sha(p)) for p in files])
    for path, obj in [('current_candidate_ms_evidence.json', output), ('CURRENT_CANDIDATE_MS_AUDIT.json', audit)]:
        (M / path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['roster', 'exact_S9_candidates', 'S9_rows', 'S9_HLA_detected_rows']}))


if __name__ == '__main__':
    main()
