"""Plot same-cohort published baselines and the calibration/order boundaries."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

P=Path(__file__).resolve().parent
O=P/'pap_neural_comparison'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
teal,purple,grey,orange='#20758a','#77559b','#737e86','#b85b32'

def style(ax):
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',color='#dce3e8',lw=.7)
    ax.tick_params(axis='y',length=0)

def main():
    native=pd.read_csv(P/'published_predictor_comparison/published_baseline_metrics.csv')
    native=native[native.cohort.eq('Voutsinos_primary60')].set_index('model')
    neural=pd.read_csv(O/'native_metrics.csv')
    neural=neural[neural.cohort.eq('Voutsinos_primary60')].set_index('model')
    effect=pd.read_csv(P/'independent_interventions/transfer_metrics.csv')
    effect=effect[effect.cohort.eq('primary60_common')].set_index('model')
    ne=pd.read_csv(O/'paired_effect_metrics.csv')
    ne=ne[ne.cohort.eq('Chu_primary60_common')].set_index('model')
    left=[('QCDPred: mean 17-mer windows',native.loc['QCDPred_mean17_stability','rho'],grey),
        ('QCDPred: whole composition',native.loc['QCDPred_normalized_cargo_stability','rho'],grey),
        ('GFP-trained A',native.loc['GFP_MP__A','rho'],teal),('GFP-trained T',native.loc['GFP_MP__T','rho'],teal),
        ('mCherry-trained A',native.loc['MP_mCherry__A','rho'],orange),('mCherry-trained T',native.loc['MP_mCherry__T','rho'],orange),
        ('PAP CNN1: same-study ref.',neural.loc['cnn2w1__C30__full','rho'],purple),
        ('PAP CNN7: same-study ref.',neural.loc['cnn2w7__C30__full','rho'],purple)]
    right=[('GFP-trained A',effect.loc['GFP_MP__A','delta_rho'],teal),
        ('GFP-trained T',effect.loc['GFP_MP__T','delta_rho'],teal),
        ('QCDPred: whole composition',effect.loc['QCDPred_normalized_cargo','delta_rho'],grey),
        ('Linear PAP: full 58 aa',effect.loc['PAP_full_cargo','delta_rho'],grey),
        ('Linear PAP: full 58 aa + WT map',effect.loc['PAP_full_cargo__source_WT_calibrated','delta_rho'],grey),
        ('PAP CNN1: last 30 aa',ne.loc['cnn2w1__C30__full','delta_rho'],purple),
        ('PAP CNN1: full 58 aa',ne.loc['cnn2w1__full_cargo__full','delta_rho'],purple),
        ('PAP CNN1: full 58 aa + WT map',ne.loc['cnn2w1__full_cargo__full__source_WT_calibrated','delta_rho'],purple),
        ('PAP CNN7: full 58 aa',ne.loc['cnn2w7__full_cargo__full','delta_rho'],purple)]
    fig,axes=plt.subplots(1,2,figsize=(12,6.2))
    for ax,rows,title,limits in [(axes[0],left,'A  Native 30-mer ranking',(.60,.925)),
                                (axes[1],right,'B  Paired 58-mer changes',(.48,.79))]:
        for i,(label,value,color) in enumerate(rows):
            ax.scatter(value,i,c=color,s=45,marker='D' if 'WT map' in label else 'o',zorder=3)
            ax.text(value+.008,i,f'{value:.3f}',va='center',fontsize=9)
        ax.set_yticks(range(len(rows)),[r[0] for r in rows]);ax.set_ylim(len(rows)-.45,-.65)
        ax.set_xlim(limits);ax.set_xlabel('Spearman correlation');ax.set_title(title,loc='left',pad=15)
        style(ax)
    axes[0].axhline(5.5,color=grey,ls='--',lw=.8)
    fig.subplots_adjust(left=.23,right=.98,bottom=.25,top=.88,wspace=1.55)
    fig.text(.03,.14,'A:14,143 tiles/383 genes. PAP native references were trained in the target study.',fontsize=9.5)
    fig.text(.03,.095,'B:1,743 independent pairs/1,591 groups; PAP-training-linked genes/windows excluded.',fontsize=9.5)
    fig.text(.03,.05,'Full 58 aa is variable-length inference beyond 30-mer training length. No target fitting; WT maps use Guo source only.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(O/f'published_baseline_challenge.{ext}',dpi=300)
    plt.close(fig)
    cal=pd.read_csv(O/'paired_effect_contrasts.csv')
    native_con=pd.read_csv(O/'native_order_contrasts.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    cal=cal[cal.role.eq('source-only calibration')]
    for i,row in enumerate(cal.itertuples()):
        axes[0].errorbar(row.estimate,i,xerr=[[row.estimate-row.low],[row.high-row.estimate]],fmt='o',color=purple,capsize=3)
    axes[0].set_yticks(range(4),['CNN1 last 30 aa','CNN1 full 58 aa','CNN7 last 30 aa','CNN7 full 58 aa'])
    axes[0].set_xlim(-.012,.025);axes[0].set_ylim(3.5,-.6)
    axes[0].set_title('A  Source WT calibration gain',loc='left',pad=15)
    rows=[native_con.iloc[0],pd.read_csv(O/'paired_effect_contrasts.csv').query("role == 'order outside fixed ends'").iloc[0],
          pd.read_csv(O/'paired_effect_contrasts.csv').query("role == 'order outside fixed ends'").iloc[1]]
    for i,row in enumerate(rows):
        axes[1].errorbar(row.estimate,i,xerr=[[row.estimate-row.low],[row.high-row.estimate]],fmt='o',color=teal,capsize=3)
    axes[1].set_yticks(range(3),['Native 30-mer*','Paired change: last 30 aa','Paired change: full 58 aa'])
    axes[1].set_xticks([-.01,0,.01])
    axes[1].set_xlim(-.014,.017);axes[1].set_ylim(2.5,-.6)
    axes[1].set_title('B  CNN7 native minus shuffle mean',loc='left',pad=15)
    for ax in axes:
        ax.axvline(0,color=grey,lw=.9,ls='--');ax.set_xlabel('Difference in Spearman correlation');style(ax)
    fig.subplots_adjust(left=.17,right=.97,bottom=.30,top=.84,wspace=1.25)
    fig.text(.03,.16,'Intervals:1,000 paired group resamples, conditional on fitted predictions; no multiplicity adjustment.',fontsize=9)
    fig.text(.03,.105,'Shuffles preserve composition and first/final five residues. Second shuffle seed is retained in supporting data.',fontsize=9)
    fig.text(.03,.05,'*Author-held-out target-study reference. Model-shuffle contrasts do not measure biological shuffle effects.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(O/f'calibration_and_order_boundaries.{ext}',dpi=300)
    plt.close(fig)
    print('Saved two scientific comparison figures.')

if __name__=='__main__':main()
