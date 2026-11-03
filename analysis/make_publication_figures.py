"""Redraw frozen results for the SSRN preprint; no fitting or inference."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator, FixedLocator
import pymupdf

P=Path(__file__).resolve().parent
S=P/'figures/source_data'
F=P/'figures'
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; PURPLE='#8B5AA8'; GREY='#747C83'
W=160/25.4
OUTPUTS=[]; CHECKS=[]; PLOTTED=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(n):return pd.read_csv(S/n)
def setup():
    font_manager.findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,
        'axes.titlesize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'legend.fontsize':6.5,'axes.linewidth':.5,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':False,'xtick.major.width':.5,
        'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5,
        'lines.linewidth':.8,'lines.markersize':3,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':450,
        'text.color':'black','axes.labelcolor':'black','xtick.color':'black',
        'ytick.color':'black','figure.facecolor':'white','axes.facecolor':'white'})

def panels(fig,axes,titles):
    for i,(ax,title) in enumerate(zip(axes,titles)):
        b=ax.get_position()
        fig.text(b.x0-.035,b.y1+.035,chr(97+i),weight='bold',size=8,va='bottom')
        ax.set_title(title,loc='left',pad=9,weight='normal')

def interval(ax,x,y,lo,hi,color=BLUE,marker='o',label=None):
    ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,
                markersize=3,capsize=2,elinewidth=.7,capthick=.7,label=label,zorder=3)

def record(figid,panel,label,value,lo=None,hi=None):
    PLOTTED.append(dict(figure=figid,panel=panel,label=label,value=float(value),
        low=None if lo is None else float(lo),high=None if hi is None else float(hi)))

def save(fig,name):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();fb=fig.bbox
    outside=[];unused_tick_labels=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if tick.get_loc()<lo-1e-9 or tick.get_loc()>hi+1e-9:
                    unused_tick_labels.update([id(tick.label1),id(tick.label2)])
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in unused_tick_labels:continue
        b=t.get_window_extent(renderer)
        if b.x0<-.8 or b.y0<-.8 or b.x1>fb.x1+.8 or b.y1>fb.y1+.8:
            outside.append(t.get_text())
    assert not outside,(name,outside)
    sizes=[t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
        if t.get_visible() and t.get_text().strip()]
    assert min(sizes)>=5 and max(sizes)<=8,(name,min(sizes),max(sizes))
    for ext in ['pdf','svg','png']:
        p=F/f'{name}.{ext}'
        fig.savefig(p,dpi=450,facecolor='white')
        OUTPUTS.append(dict(path=p.relative_to(P).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    with pymupdf.open(F/f'{name}.pdf') as d:
        assert len(d)==1 and not d[0].get_images()
        for block in d[0].get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line.get('spans',[]):
                    b=pymupdf.Rect(span['bbox']);r=d[0].rect
                    assert b.x0>=-.5 and b.y0>=-.5 and b.x1<=r.width+.5 and b.y1<=r.height+.5,(name,span)
        fonts=d[0].get_fonts(full=True)
        assert fonts and all(f[1] for f in fonts),fonts
        mm=[v*25.4/72 for v in d[0].rect[2:]]
        assert abs(mm[0]-160)<.1 and mm[1]<=170
    CHECKS.append(dict(figure=name,width_mm=mm[0],height_mm=mm[1],min_text_pt=min(sizes),
        max_text_pt=max(sizes),out_of_canvas_text=[],embedded_fonts=[f[3] for f in fonts],
        vector_pdf_no_embedded_raster=True))
    plt.close(fig)

def figure1():
    fig,ax=plt.subplots(1,2,figsize=(W,80/25.4))
    fig.subplots_adjust(left=.20,right=.98,bottom=.19,top=.84,wspace=.55)
    labels=['Guo WT / GFP–MP','Additional Guo','Yang FLAG panel']
    total=np.array([1072,801,37]);strict=np.array([1072,602,20]);y=np.arange(3)
    ax[0].barh(y,total,color='#DDE2E5',height=.56,label='Available')
    ax[0].barh(y,strict,color=BLUE,height=.56,label='Analysis / strict')
    ax[0].set(yticks=y,yticklabels=labels,xlim=(0,1430),xlabel='Measured constructs')
    ax[0].invert_yaxis();ax[0].set_ylim(2.85,-.65)
    for i,(n,m) in enumerate(zip(strict,total)):
        ax[0].text(m+30,i,f'{n:,} / {m:,}',va='center',size=6)
        record('Fig1','a',labels[i],n)
    ax[0].legend(frameon=False,loc='lower right',handlelength=1.2)
    g=csv('development_group_sizes.csv')['constructs'].to_numpy()
    assert len(g)==800 and g.max()==156 and g.sum()==1072
    ax[1].plot(np.arange(1,len(g)+1),g,color=BLUE)
    ax[1].set(yscale='log',xlabel='Group rank by size',ylabel='WT constructs in group',xlim=(0,820))
    ax[1].text(.37,.73,'Largest group: 156\n5 outer / 3 inner folds\n261 cargos ≤30 aa',
        transform=ax[1].transAxes,size=6.2,linespacing=1.4)
    panels(fig,ax,['Endpoint coverage','800 dependence groups']);save(fig,'Fig1')

def dots(ax,cohort,models,labels,figid,panel):
    q=csv('benchmark_conditional_intervals.csv').query('cohort == @cohort').set_index('model').loc[models]
    for i,(k,r) in enumerate(q.iterrows()):
        interval(ax,r.rho,i,r.rho_low,r.rho_high)
        record(figid,panel,k,r.rho,r.rho_low,r.rho_high)
    ax.set(yticks=range(len(labels)),yticklabels=labels,xlim=(-1,1.04),ylim=(len(labels)-.45,-.55),
           xlabel='Spearman correlation')
    ax.xaxis.set_major_locator(FixedLocator([-1,-.5,0,.5,1]))
    ax.axvline(0,color=GREY,lw=.6,ls='--',zorder=0)

def figure2():
    fig,ax=plt.subplots(1,2,figsize=(W,94/25.4))
    fig.subplots_adjust(left=.19,right=.98,bottom=.16,top=.86,wspace=.95)
    dots(ax[0],'native_GFP_MP',['AAC_resemblance','ShortStop','PAP_Cterminal','PAP_full','A','T','O'],
        ['AAC resemblance','ShortStop (1,057)','PAP C30','PAP full (secondary)','A: composition','T: + terminals','O: + native order'],'Fig2','a')
    dots(ax[1],'additional_strict',['AAC_resemblance','ShortStop','PAP_Cterminal','A','T','O','Guo_regression'],
        ['AAC resemblance','ShortStop (595)','PAP C30','A: composition','T: + terminals','O: + native order','Guo regression'],'Fig2','b')
    panels(fig,ax,['WT reporter ranking (n = 1,072)','Additional transfer (n = 602)']);save(fig,'Fig2')

def figure3():
    z=pd.DataFrame(json.loads((S/'native_contrasts.json').read_text(encoding='utf-8')))
    names=['Primary GFP–MP','MP–mCherry','Mutagenesis WT','40% identity grouping','Fold seed +1','Fold seed +2']
    fig,ax=plt.subplots(1,2,figsize=(W,86/25.4))
    fig.subplots_adjust(left=.22,right=.98,bottom=.20,top=.85,wspace=1.0)
    for a,key,xlabel,p in [(ax[0],'rank_gain','Order − terminal rank gain','a'),
                           (ax[1],'mae_reduction','Terminal − order MAE','b')]:
        for i,r in z.iterrows():
            lo,hi=r[key+'_CI'];interval(a,r[key],i,lo,hi);record('Fig3',p,names[i],r[key],lo,hi)
        a.set(yticks=range(6),yticklabels=names,ylim=(5.5,-.5),xlabel=xlabel)
        a.axvline(0,color=GREY,lw=.6,ls='--');a.xaxis.set_major_locator(MaxNLocator(4))
    panels(fig,ax,['Incremental residue order','Prediction-error reduction']);save(fig,'Fig3')

def figure4():
    d=csv('mutation_mae.csv')
    labs=['Family mean','WT-trained T','WT-trained O','Calibrated PAP C30','Calibrated PAP full',
          'Delta-trained T','Opposite-family T']
    fig,ax=plt.subplots(1,2,figsize=(W,92/25.4))
    fig.subplots_adjust(left=.22,right=.98,bottom=.18,top=.86,wspace=1.0)
    for a,fam,p in [(ax[0],'L_to_A','a'),(ax[1],'microprotein_10mer_ext','b')]:
        q=d[d.family.eq(fam)]
        assert len(q)==7
        colors=[GREY,BLUE,BLUE,GREEN,GREEN,ORANGE,ORANGE]
        a.barh(range(7),q.mae,color=colors,height=.58)
        a.set(yticks=range(7),yticklabels=labs,ylim=(6.55,-.55),xlim=(0,.19),
            xlabel='Mean absolute error of score change')
        for i,r in enumerate(q.itertuples()):
            a.text(r.mae+.003,i,f'{r.mae:.3f}',va='center',size=6)
            record('Fig4',p,r.model,r.mae)
    panels(fig,ax,['Leu-to-Ala (n = 1,014)','C-terminal append (n = 1,007)']);save(fig,'Fig4')

def figure5():
    fig,ax=plt.subplots(1,2,figsize=(W,94/25.4),gridspec_kw={'width_ratios':[1.2,1]})
    fig.subplots_adjust(left=.25,right=.98,bottom=.17,top=.86,wspace=.55)
    dots(ax[0],'Yang_strict',['published_PepScore','Length','PAP_Cterminal_fusion_sequence',
        'PAP_full_fusion_sequence','MP_mCherry_T_fusion_sequence','MP_mCherry_O_fusion_sequence','GFP_MP_T_native_sequence'],
        ['Published PepScore','Native length','PAP C30 + FLAG','PAP full + FLAG','mCherry T + FLAG',
         'mCherry O + FLAG','GFP T, native'],'Fig5','a')
    v=csv('yang_outcomes.csv');mask=v.strict_disjoint
    ax[1].scatter(v.loc[~mask,'native_length'],v.loc[~mask,'untreated_mean'],s=13,
        facecolors='none',edgecolors=GREY,linewidths=.65,label='Other 17')
    ax[1].scatter(v.loc[mask,'native_length'],v.loc[mask,'untreated_mean'],s=13,
        c=BLUE,linewidths=.3,label='Strict 20')
    ax[1].axhline(.05,color=GREY,lw=.6,ls='--')
    ax[1].set(xlabel='Native peptide length (aa)',ylabel='Normalized FLAG-positive fraction')
    ax[1].legend(frameon=False,loc='center left',handletextpad=.4)
    panels(fig,ax,['Separate study (strict n = 20)','Assay and selection context']);save(fig,'Fig5')

def figure6():
    d=csv('external_metrics.csv').query("subset == 'primary60'")
    pol=csv('top_decile_prioritization.csv')
    names=['A','T','T_noPAP','N4','C4','I4']
    fig,ax=plt.subplots(1,2,figsize=(W,86/25.4),gridspec_kw={'width_ratios':[1.35,1]})
    fig.subplots_adjust(left=.09,right=.98,bottom=.29,top=.85,wspace=.45)
    for source,off,col,marker,label in [('GFP_MP',-.13,BLUE,'o','GFP–MP training'),
        ('MP_mCherry',.13,ORANGE,'s','MP–mCherry training')]:
        for i,k in enumerate(names):
            r=d[d.model.eq(source+'__'+k)].iloc[0]
            ax[0].plot(i+off,r.rho,marker=marker,color=col,ls='none',label=label if i==0 else None)
            if i<2:ax[0].errorbar(i+off,r.rho,yerr=[[r.rho-r.low],[r.high-r.rho]],fmt='none',
                color=col,capsize=2,lw=.7)
            record('Fig6','a',source+'__'+k,r.rho,r.low if i<2 else None,r.high if i<2 else None)
        vals=[float(pol[pol.model.eq(source+'__'+k)].selected_mean_assay_score.iloc[0]) for k in ['A','T']]
        ax[1].plot(np.arange(2)+off,vals,marker=marker,color=col)
        for k,val in zip(['A','T'],vals):record('Fig6','b',source+'__'+k,val)
    pap=float(d[d.model.eq('PAP_full')].rho.iloc[0])
    ax[0].axhline(pap,color=GREY,lw=.7,ls='--')
    ax[0].text(.98,.94,'PAP: target-study reference',transform=ax[0].transAxes,ha='right',size=6)
    ax[0].set(xticks=range(6),xticklabels=['A','T','T without\nPAP','N4','C4','I4'],
        ylabel='Spearman correlation',ylim=(.6,.89),xlim=(-.5,5.5))
    ax[1].set(xticks=[0,1],xticklabels=['A: composition','T: + terminals'],xlim=(-.35,1.35),
        ylim=(.52,.64),ylabel='Mean selected assay score')
    handles,labels=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.54,.025))
    panels(fig,ax,['Independent 30-mer ranking','Top-decile selection']);save(fig,'Fig6')

def figure7():
    n=csv('published_baseline_metrics.csv').query("cohort == 'Voutsinos_primary60'").set_index('model')
    nn=csv('neural_native_metrics.csv').query("cohort == 'Voutsinos_primary60'").set_index('model')
    e=csv('intervention_metrics.csv').query("cohort == 'primary60_common'").set_index('model')
    ne=csv('neural_paired_metrics.csv').query("cohort == 'Chu_primary60_common'").set_index('model')
    left=[('QCDPred: mean 17-mer windows',n.loc['QCDPred_mean17_stability','rho'],GREY),
        ('QCDPred: whole composition',n.loc['QCDPred_normalized_cargo_stability','rho'],GREY),
        ('GFP-trained A',n.loc['GFP_MP__A','rho'],BLUE),('GFP-trained T',n.loc['GFP_MP__T','rho'],BLUE),
        ('mCherry-trained A',n.loc['MP_mCherry__A','rho'],ORANGE),('mCherry-trained T',n.loc['MP_mCherry__T','rho'],ORANGE),
        ('PAP CNN1: same-study reference',nn.loc['cnn2w1__C30__full','rho'],PURPLE),
        ('PAP CNN7: same-study reference',nn.loc['cnn2w7__C30__full','rho'],PURPLE)]
    right=[('GFP-trained A',e.loc['GFP_MP__A','delta_rho'],BLUE),('GFP-trained T',e.loc['GFP_MP__T','delta_rho'],BLUE),
        ('QCDPred: whole composition',e.loc['QCDPred_normalized_cargo','delta_rho'],GREY),
        ('Linear PAP: full 58 aa',e.loc['PAP_full_cargo','delta_rho'],GREY),
        ('Linear PAP: full 58 aa + WT map',e.loc['PAP_full_cargo__source_WT_calibrated','delta_rho'],GREY),
        ('PAP CNN1: last 30 aa',ne.loc['cnn2w1__C30__full','delta_rho'],PURPLE),
        ('PAP CNN1: full 58 aa',ne.loc['cnn2w1__full_cargo__full','delta_rho'],PURPLE),
        ('PAP CNN1: full 58 aa + WT map',ne.loc['cnn2w1__full_cargo__full__source_WT_calibrated','delta_rho'],PURPLE),
        ('PAP CNN7: full 58 aa',ne.loc['cnn2w7__full_cargo__full','delta_rho'],PURPLE)]
    fig,ax=plt.subplots(1,2,figsize=(W,106/25.4))
    fig.subplots_adjust(left=.27,right=.98,bottom=.14,top=.87,wspace=1.8)
    for a,rows,p,lim in [(ax[0],left,'a',(.60,.935)),(ax[1],right,'b',(.48,.80))]:
        for i,(label,val,col) in enumerate(rows):
            a.plot(val,i,'D' if 'WT map' in label else 'o',color=col,markersize=3)
            a.annotate(f'{val:.3f}',(val,i),xytext=(4,0),textcoords='offset points',va='center',size=5.5)
            record('Fig7',p,label,val)
        a.set(yticks=range(len(rows)),yticklabels=[x[0] for x in rows],
            ylim=(len(rows)-.5,-.6),xlim=lim,xlabel='Spearman correlation')
        a.xaxis.set_major_locator(MaxNLocator(4))
    ax[0].axhline(5.5,color=GREY,ls='--',lw=.6)
    panels(fig,ax,['Native 30-mers (n = 14,143)','Paired changes (n = 1,743)']);save(fig,'Fig7')

def figure8():
    d=csv('neural_paired_contrasts.csv');cal=d[d.role.eq('source-only calibration')]
    native=csv('native_order_contrasts.csv');order=d[d.role.eq('order outside fixed ends')]
    fig,ax=plt.subplots(1,2,figsize=(W,85/25.4))
    fig.subplots_adjust(left=.20,right=.98,bottom=.20,top=.85,wspace=1.15)
    labels=['CNN1 last 30 aa','CNN1 full 58 aa','CNN7 last 30 aa','CNN7 full 58 aa']
    for i,r in enumerate(cal.itertuples()):
        interval(ax[0],r.estimate,i,r.low,r.high,color=PURPLE)
        record('Fig8','a',r.contrast,r.estimate,r.low,r.high)
    ax[0].set(yticks=range(4),yticklabels=labels,ylim=(3.6,-.6),xlim=(-.012,.025))
    rows=[native.iloc[0],order.iloc[0],order.iloc[1]]
    for i,r in enumerate(rows):
        interval(ax[1],r.estimate,i,r.low,r.high);record('Fig8','b',r.contrast,r.estimate,r.low,r.high)
    ax[1].set(yticks=range(3),yticklabels=['Native 30-mers','Paired: last 30 aa','Paired: full 58 aa'],
        ylim=(2.55,-.55),xlim=(-.014,.017))
    for a in ax:
        a.axvline(0,color=GREY,lw=.6,ls='--');a.set_xlabel('Difference in Spearman correlation')
        a.xaxis.set_major_locator(MaxNLocator(4))
    panels(fig,ax,['Source-WT calibration gain','CNN7 native − shuffle mean']);save(fig,'Fig8')

def figure9():
    d=pd.concat([csv('terminal_transfer_metrics.csv'),csv('terminal_window_metrics.csv')])
    cohorts=['all1743_provenance_limited','neither_endpoint_screen_matched','neither_endpoint_or_available_symbol_overlap']
    labels=['All pairs\n1,743 / 1,591 groups','Neither endpoint matched\n289 / 259 groups',
        'Also no symbol match\n97 / 88 groups']
    models=[('DEGRONOPEDIA_C23','DEGRONOPEDIA C23',ORANGE,'o'),
        ('cnn2w1__C23__full','PAP CNN1 C23',BLUE,'s'),
        ('cnn2w1__C30__full','PAP CNN1 C30',GREEN,'^'),
        ('cnn2w1__full_cargo__full','PAP CNN1 full 58 aa',PURPLE,'D')]
    fig,ax=plt.subplots(1,2,figsize=(W,103/25.4),gridspec_kw={'width_ratios':[1.3,1]})
    fig.subplots_adjust(left=.25,right=.98,bottom=.28,top=.85,wspace=.70)
    for j,(key,label,col,marker) in enumerate(models):
        for i,c in enumerate(cohorts):
            v=float(d[d.cohort.eq(c)&d.model.eq(key)].delta_rho.iloc[0]);y=i+(j-1.5)*.16
            ax[0].plot(v,y,marker=marker,color=col,ls='none',label=label if i==0 else None)
            ax[0].text(v+.009,y,f'{v:.3f}',va='center',size=5.5)
            record('Fig9','a',c+' / '+key,v)
    ax[0].set(yticks=range(3),yticklabels=labels,ylim=(2.5,-.5),xlim=(.4,.86),xlabel='Spearman correlation')
    contrasts=csv('terminal_window_contrasts.csv')
    for i,c in enumerate(cohorts):
        r=contrasts[contrasts.cohort.eq(c)&contrasts.contrast.str.contains('__C23__')].iloc[0]
        interval(ax[1],r.estimate,i,r.low,r.high);record('Fig9','b',c,r.estimate,r.low,r.high)
    ax[1].axvline(0,color=GREY,lw=.6,ls='--')
    ax[1].set(yticks=range(3),yticklabels=['All 1,743','Unmatched 289','Symbol sensitivity 97'],
        ylim=(2.5,-.5),xlim=(-.09,.14),xlabel='Difference in Spearman correlation')
    ax[1].xaxis.set_major_locator(MaxNLocator(4))
    handles,labs=ax[0].get_legend_handles_labels()
    fig.legend(handles,labs,ncol=2,loc='lower center',bbox_to_anchor=(.55,.025),frameon=False)
    panels(fig,ax,['Paired-change ranking','Same C23: C-model − CNN1']);save(fig,'Fig9')

def supplementary1():
    t=csv('clustering_display_coordinates.csv');m=csv('cluster_metrics.csv')
    fig,ax=plt.subplots(2,2,figsize=(W,128/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.11,top=.91,hspace=.66,wspace=.38)
    for cl,col in [('nORF',ORANGE),('canonical',BLUE)]:
        z=t[t['class'].eq(cl)]
        ax[0,0].scatter(z.tsne1,z.tsne2,s=2,alpha=.4,c=col,lw=0,label=cl)
        ax[0,1].scatter(z.pc1,z.pc2,s=2,alpha=.4,c=col,lw=0)
    var=json.loads((S/'pca_variance.json').read_text(encoding='utf-8'))
    ax[0,0].set(xlabel='t-SNE 1',ylabel='t-SNE 2')
    ax[0,0].legend(frameon=False,markerscale=2,handletextpad=.3)
    ax[0,1].set(xlabel=f'PC1 ({var[0]:.1%})',ylabel=f'PC2 ({var[1]:.1%})')
    for scheme,ls in [('per_feature','-'),('per_family','--')]:
        q=m[m.scheme.eq(scheme)&m.space.eq('PCA50')].groupby('k').mean(numeric_only=True)
        for kind,col in [('native',ORANGE),('shuffle',BLUE)]:
            ax[1,0].plot(q.index,q['silhouette_'+kind],ls=ls,marker='o',color=col,
                label=scheme.replace('per_','')+' / '+kind)
        for key,marker,col,label in [('label_ARI','o',ORANGE,'labels'),('native_shuffle_ARI','s',BLUE,'native–shuffle')]:
            ax[1,1].plot(q.index,q[key],ls=ls,marker=marker,color=col,label=scheme.replace('per_','')+' / '+label)
    for a,y in [(ax[1,0],'Silhouette'),(ax[1,1],'Adjusted Rand index')]:
        a.set(xlabel='Number of clusters, k',ylabel=y);a.legend(frameon=False,fontsize=5.5,handlelength=2)
    panels(fig,list(ax.flat),['Display-only embedding','Linear representation',
        'Native and shuffle clustering','Cluster agreement']);save(fig,'S1_Fig')

def supplementary2():
    obs=csv('paired_observed_agreement.csv');d=csv('paired_transfer_metrics.csv').query("setting == 'primary'")
    fig,ax=plt.subplots(1,2,figsize=(W,96/25.4))
    fig.subplots_adjust(left=.23,right=.98,bottom=.25,top=.85,wspace=.75)
    labs=['G vs M','G vs U','M vs U','G replicates','M replicates','U replicates']
    for i,r in enumerate(obs.itertuples()):interval(ax[0],r.rho,i,r.rho_low,r.rho_high,color=BLUE if i<3 else GREY)
    ax[0].set(yticks=range(6),yticklabels=labs,ylim=(5.5,-.5),xlim=(.3,1.03),xlabel='Spearman correlation')
    ax[0].axhline(2.5,color=GREY,lw=.5)
    names=['GFP_original','mCherry_original','GFP_mutagenesis'];short=dict(zip(names,'GMU'))
    pairs=[(s,t) for s in names for t in names if s!=t]
    for method,model,off,col,marker,label in [('source_sequence_raw','A',-.18,GREEN,'^','Source A'),
        ('source_sequence_raw','T',0,ORANGE,'o','Source T'),('target_sequence','T',.18,BLUE,'s','Target T')]:
        for i,(s,t) in enumerate(pairs):
            r=d[d.source.eq(s)&d.target.eq(t)&d.method.eq(method)&d.model.eq(model)].iloc[0]
            interval(ax[1],r.rho,i+off,r.rho_low,r.rho_high,color=col,marker=marker,label=label if i==0 else None)
            record('S2_Fig','b',s+' / '+t+' / '+label,r.rho,r.rho_low,r.rho_high)
    ax[1].set(yticks=range(6),yticklabels=[short[s]+' → '+short[t] for s,t in pairs],
        ylim=(5.5,-.5),xlim=(.35,.88),xlabel='Held-out target rank correlation')
    h,l=ax[1].get_legend_handles_labels()
    fig.legend(h,l,ncol=3,loc='lower center',bbox_to_anchor=(.64,.035),frameon=False)
    panels(fig,ax,['Same-peptide agreement','Sequence-model transfer']);save(fig,'S2_Fig')

def main():
    sys.stdout.reconfigure(encoding='utf-8');setup()
    for f in [figure1,figure2,figure3,figure4,figure5,figure6,figure7,figure8,figure9,supplementary1,supplementary2]:f()
    (F/'PLOTTED_VALUES.json').write_text(json.dumps(PLOTTED,indent=2)+'\n',encoding='utf-8')
    (F/'FIGURE_TECHNICAL_QA.json').write_text(json.dumps(dict(figures=CHECKS,files=OUTPUTS,
        style='Arial 5.5–7 pt; lowercase 8 pt panels; vector PDF/SVG; accessible colours; no decorative grids',
        new_fit=False,values_from_frozen_sources=True,visual_qa_complete=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(figures=len(CHECKS),exports=len(OUTPUTS),plotted_values=len(PLOTTED),technical_checks='passed')))

if __name__=='__main__':main()
