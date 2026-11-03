"""Scientific figure for the independently evaluated transfer result."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
O = P / 'independent_test'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':11,
    'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def main():
    metric = pd.read_csv(O/'external_metrics.csv').query("subset == 'primary60'")
    policy = pd.read_csv(O/'top_decile_prioritization.csv')
    fig, (ax,bx) = plt.subplots(1,2,figsize=(8.5,4.2),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.085,right=.98,bottom=.27,top=.84,wspace=.38)
    names = ['A','T','T_noPAP','N4','C4','I4']
    for source,offset,color,label in [('GFP_MP',-.12,'#277285','GFP-MP training'),
                                      ('MP_mCherry',.12,'#b05a32','MP-mCherry training')]:
        x = np.arange(len(names))+offset
        rows = [metric[metric.model.eq(source+'__'+n)].iloc[0] for n in names]
        yy = np.array([r.rho for r in rows])
        ax.plot(x,yy,'o-',color=color,lw=1.1,markersize=4,label=label)
        for j in [0,1]:
            r = rows[j]
            ax.errorbar(x[j],r.rho,yerr=[[r.rho-r.low],[r.high-r.rho]],fmt='none',color=color,capsize=2,lw=1)
        yy = [float(policy[policy.model.eq(source+'__'+m)].selected_mean_assay_score.iloc[0]) for m in ['A','T']]
        xx = np.array([0,1])+offset
        bx.plot(xx, yy, 'o-',color=color,lw=1.1,markersize=5)
    pap = metric[metric.model.eq('PAP_full')].rho.iloc[0]
    ax.axhline(pap,color='#7d858c',ls='--',lw=1)
    ax.text(2.6,pap+.004,'PAP: trained in target study',fontsize=7.4,color='#636d73')
    ax.set_xticks(range(len(names)),['A','T','T without\nPAP','N4','C4','I4'])
    ax.set_ylabel('External Spearman correlation')
    ax.set_ylim(.60,.89)
    ax.set_title('A  Transfer to independent peptide assay',loc='left',pad=16)
    ax.grid(axis='y',lw=.5,color='#e2e7ea');ax.set_axisbelow(True)
    bx.set_xticks([0,1],['A: composition','T: terminals added'])
    bx.set_ylabel('Mean assay score of selected peptides')
    bx.set_ylim(.52,.64)
    bx.set_title('B  Top-decile selection',loc='left',pad=16)
    bx.grid(axis='y',lw=.5,color='#e2e7ea');bx.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.105),frameon=False)
    fig.text(.085,.035,'14,143 canonical 30-mers; 383 gene groups. Primary A/T intervals: conditional gene bootstrap.',fontsize=8)
    fig.text(.085,.005,'N4/C4/I4: equal-size positional diagnostics. Canonical tiles do not validate endogenous nORF function.',fontsize=8)
    fig.savefig(O/'independent_transfer_and_prioritization.png',dpi=300,facecolor='white')
    fig.savefig(O/'independent_transfer_and_prioritization.svg',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
