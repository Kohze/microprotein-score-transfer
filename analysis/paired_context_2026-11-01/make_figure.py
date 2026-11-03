"""Paired contexts: observed agreement and all six directed transfers."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
R = P / 'results'
F = P / 'figures'
F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.5,
                     'axes.labelsize': 9, 'axes.titlesize': 10,
                     'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})
LABEL = dict(GFP_original='GFP original', mCherry_original='mCherry original',
             GFP_mutagenesis='GFP mutation library')


def main():
    obs = pd.read_csv(R / 'observed_agreement.csv')
    m = pd.read_csv(R / 'transfer_metrics.csv')
    m = m[m.setting.eq('primary')]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.35), gridspec_kw={'width_ratios': [1, 1.25]})
    fig.subplots_adjust(left=.04, right=.985, bottom=.18, top=.88, wspace=.56)
    ax = axes[0]
    labels = ['GFP vs mCherry', 'GFP original vs\nGFP mutation library',
              'mCherry vs\nGFP mutation library', 'GFP original', 'mCherry original', 'GFP mutation library']
    for i, r in enumerate(obs.itertuples()):
        color = '#277285' if i < 3 else '#969da5'
        ax.errorbar(r.rho, i, xerr=[[r.rho-r.rho_low], [r.rho_high-r.rho]],
                    fmt='o', markersize=4, color=color, capsize=2, lw=1)
    ax.set_yticks(range(6), labels, ha='left')
    ax.tick_params(axis='y', pad=0, length=0)
    ax.set_xlim(.3, 1.03)
    # Put labels outside and give the whole figure a clear text gutter.
    for tick in ax.get_yticklabels():
        tick.set_horizontalalignment('right')
    ax.invert_yaxis()
    ax.axhline(2.5, color='#cccccc', lw=.6)
    ax.set_xlabel('Spearman correlation')
    ax.set_title('A  Same-peptide agreement', loc='left', pad=26)
    ax.text(.02, 1.025, 'Between contexts / within-context replicates', transform=ax.transAxes, fontsize=7.5)
    # The left text gutter is accounted for explicitly below.
    ax = axes[1]
    pairs = [(s, t) for s in LABEL for t in LABEL if s != t]
    short = dict(GFP_original='G', mCherry_original='M', GFP_mutagenesis='U')
    for method, model, offset, color, marker, label in [
        ('source_sequence_raw', 'A', -.18, '#72943c', '^', 'Source A'),
        ('source_sequence_raw', 'T', 0, '#b05a32', 'o', 'Source T'),
        ('target_sequence', 'T', .18, '#277285', 's', 'Target T')]:
        for i, (s, t) in enumerate(pairs):
            r = m[m.source.eq(s) & m.target.eq(t) & m.method.eq(method) & m.model.eq(model)].iloc[0]
            ax.errorbar(r.rho, i+offset, xerr=[[r.rho-r.rho_low], [r.rho_high-r.rho]],
                        fmt=marker, markersize=4, color=color, capsize=2, lw=1,
                        label=label if i == 0 else None)
    ax.set_yticks(range(6), [short[s] + ' to ' + short[t] for s, t in pairs])
    ax.invert_yaxis()
    ax.set_xlim(.35, .88)
    ax.set_xlabel('Held-out target Spearman correlation')
    ax.set_title('B  Sequence-model transport', loc='left', pad=26)
    ax.text(.0, 1.025, 'Identical test peptides and held-out groups', transform=ax.transAxes, fontsize=7.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.76, .075), ncol=3,
               frameon=False, fontsize=7.4, columnspacing=.65, handletextpad=.2)
    fig.text(.05, .035, 'G: original GFP-MP; M: MP-mCherry; U: GFP-MP mutation library.  n = 1,044; 783 groups.', fontsize=8)
    # Explicit positions preserve legible long context labels at journal width.
    axes[0].set_position([.245, .235, .25, .61])
    axes[1].set_position([.665, .235, .31, .61])
    for ax in axes:
        ax.grid(axis='x', color='#e6eaed', lw=.5)
        ax.set_axisbelow(True)
    fig.savefig(F / 'S2_Fig.png', dpi=300, facecolor='white')
    fig.savefig(F / 'S2_Fig.svg', facecolor='white')
    plt.close(fig)
    from PIL import Image
    im = Image.open(F / 'S2_Fig.png').convert('RGB')
    im.save(F / 'S2_Fig.tif', compression='tiff_lzw', dpi=(300, 300))
    (F / 'FIGURE_RECEIPT.json').write_text(json.dumps(dict(width_pixels=im.width, height_pixels=im.height,
        dpi=300, plot_source_files=['observed_agreement.csv', 'transfer_metrics.csv'],
        scope='exploratory; conditional group-bootstrap intervals; no causal tag attribution'), indent=2) + '\n')


if __name__ == '__main__':
    main()
