"""Secondary paired A/T contrasts; all fitted predictions are unchanged."""
from pathlib import Path
import pandas as pd
import analyze_paired_context as a

P = Path(__file__).resolve().parent
R = P / 'results'


def main():
    rows = []
    for setting in ['primary', 'identity40', 'seed32']:
        p = pd.read_parquet(R / (setting + '_predictions.parquet'))
        ix = a.bootstrap_indices(p.eval_group.to_numpy()) if setting == 'primary' else []
        for source in a.CONTEXTS:
            for target in a.CONTEXTS:
                y = p[target + '__observed'].to_numpy()
                aa, tt = p[source + '__A'].to_numpy(), p[source + '__T'].to_numpy()
                r = dict(setting=setting, source=source, target=target,
                         comparison='A_minus_T', rho_gap=a.rho(y, aa) - a.rho(y, tt),
                         design='secondary comparison after inspecting saved estimates')
                if ix:
                    r['low'], r['high'] = a.interval([a.rho(y[j], aa[j]) - a.rho(y[j], tt[j]) for j in ix])
                rows.append(r)
    pd.DataFrame(rows).to_csv(R / 'representation_transport_contrasts.csv', index=False)
    print(pd.DataFrame(rows).query("setting == 'primary'").to_string(index=False))


if __name__ == '__main__':
    main()
