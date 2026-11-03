"""Show the exposure and same-input limits of the published terminal challenge."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

P=Path(__file__).resolve().parent
O=P/'terminal_predictor_comparison'

def main():
    d=pd.concat([pd.read_csv(O/'transfer_metrics.csv'),pd.read_csv(O/'window_control_metrics.csv')])
    cohorts=['all1743_provenance_limited','neither_endpoint_screen_matched','neither_endpoint_or_available_symbol_overlap']
    labels=['All pairs\n1,743 / 1,591 groups','Neither endpoint matched\n289 / 259 groups','Also no available-symbol match\n97 / 88 groups']
    models=[('DEGRONOPEDIA_C23','DEGRONOPEDIA: C23','#b85b32'),('cnn2w1__C23__full','PAP CNN1: same C23','#20758a'),
            ('cnn2w1__C30__full','PAP CNN1: C30','#687a98'),('cnn2w1__full_cargo__full','PAP CNN1: full58','#77559b')]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11,5.8),gridspec_kw={'width_ratios':[1.2,1]})
    for j,(key,label,color) in enumerate(models):
        for i,cohort in enumerate(cohorts):
            value=float(d[d.cohort.eq(cohort)&d.model.eq(key)].delta_rho.iloc[0]);y=i+(j-1.5)*.16
            axes[0].scatter(value,y,color=color,s=38,label=label if i==0 else None,zorder=3)
            axes[0].text(value+.009,y,f'{value:.3f}',va='center',fontsize=8.5)
    axes[0].set_yticks(range(3),labels);axes[0].set_ylim(2.55,-.55);axes[0].set_xlim(.4,.86)
    axes[0].set_title('A  Paired-change ranking',loc='left',pad=14);axes[0].set_xlabel('Spearman correlation')
    axes[0].legend(loc='lower left',bbox_to_anchor=(-.55,-.35),ncol=2,fontsize=8.5,frameon=False)
    contrast=pd.read_csv(O/'window_control_contrasts.csv')
    for i,cohort in enumerate(cohorts):
        row=contrast[contrast.cohort.eq(cohort)&contrast.contrast.str.contains('__C23__')].iloc[0]
        axes[1].errorbar(row.estimate,i,xerr=[[row.estimate-row.low],[row.high-row.estimate]],color='#20758a',fmt='o',capsize=3)
    axes[1].axvline(0,ls='--',lw=.9,color='#737e86');axes[1].set_yticks(range(3),['All 1,743','Unmatched 289','Symbol sensitivity 97'])
    axes[1].set_ylim(2.5,-.5);axes[1].set_xlim(-.09,.14);axes[1].set_xlabel('Difference in Spearman correlation')
    axes[1].set_title('B  Same C23 input\nC-model minus CNN1',loc='left',pad=14,fontsize=11)
    for ax in axes:
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#dce3e8',lw=.7)
    fig.subplots_adjust(left=.24,right=.98,top=.87,bottom=.31,wspace=.85)
    fig.text(.03,.095,'1,454/1,743 pairs match a model development screen endpoint; the actual 90/10 training split is unavailable.',fontsize=9)
    fig.text(.03,.055,'B: post-hoc window check; 95% conditional group-bootstrap intervals. Cohort differences do not identify an exposure effect.',fontsize=9)
    fig.text(.03,.015,'The 97-pair sensitivity is not alias-complete gene independence. C23/full58 neural inputs extrapolate beyond training 30-mers.',fontsize=8.5)
    for ext in ['png','svg']:fig.savefig(O/f'terminal_transfer_exposure_and_window.{ext}',dpi=300)
    print('Saved exposure/window comparison figure.')

if __name__=='__main__':main()
