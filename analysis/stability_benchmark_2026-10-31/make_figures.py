"""Scientific figures from persisted predictions and uncertainty, with source data."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P/'results';D=P/'figures';D.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
m=pd.read_csv(R/'metrics.csv');ci=pd.read_csv(R/'conditional_intervals.csv');v=pd.read_parquet(R/'Yang_validation.parquet');n=pd.read_parquet(R/'native_GFP_MP.parquet')
blue='#25637d';orange='#bd6029';green='#3f7964';grey='#78828b'
def save(fig,name):fig.savefig(D/(name+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(D/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def panel(ax,label,title):ax.set_title(label+'  '+title,loc='left',fontweight='bold',pad=12)
def metric(cohort,model,subset='all',field='rho'):return float(m.loc[(m.cohort==cohort)&(m.model==model)&(m.subset==subset),field].iloc[0])
def dots(ax,cohort,names,labels):
    z=ci[ci.cohort.eq(cohort)].set_index('model').loc[names];y=np.arange(len(names));x=z.rho.to_numpy();ax.errorbar(x,y,xerr=[x-z.rho_low,z.rho_high-x],fmt='o',color=blue,capsize=3);ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,c=grey,lw=.8);ax.set_xlabel('Spearman correlation (95% conditional interval)');ax.set_xlim(-1,1);ax.grid(axis='x',alpha=.15);z.to_csv(D/(cohort+'_source.csv'))
fig,ax=plt.subplots(1,2,figsize=(9.2,4.2),gridspec_kw={'width_ratios':[1.05,1]})
labels=['Guo WT / GFP-MP','Additional Guo sequences','Yang FLAG constructs'];tot=[1072,801,37];strict=[1072,602,20]
y=np.arange(3);ax[0].barh(y,tot,color='#dae1e5',label='Available');ax[0].barh(y,strict,color=blue,label='Analysis / strict transfer');ax[0].set_yticks(y,labels);ax[0].invert_yaxis();ax[0].set_xlabel('Measured constructs');ax[0].legend(frameon=False,loc='lower right',fontsize=8)
for i,(a,b) in enumerate(zip(strict,tot)):ax[0].text(b+15,i,f'{a} / {b}',va='center',fontsize=9)
ax[0].set_xlim(0,1360);panel(ax[0],'A','Endpoint coverage')
g=n.groupby('eval_group').size().sort_values(ascending=False).to_numpy();ax[1].plot(np.arange(1,len(g)+1),g,c=blue);ax[1].set_yscale('log');ax[1].set_xlabel('Dependence group, sorted by size');ax[1].set_ylabel('WT constructs in group');panel(ax[1],'B','800 development groups');ax[1].text(.43,.78,'Largest group: 156\n5 outer / 3 inner folds\n261 cargos at most 30 aa',transform=ax[1].transAxes,fontsize=9)
fig.tight_layout();save(fig,'Figure1')
pd.DataFrame({'cohort':labels,'available':tot,'analysis_or_strict':strict}).to_csv(D/'Figure1_source.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(10,4.8))
dots(ax[0],'native_GFP_MP',['AAC_resemblance','ShortStop','PAP_Cterminal','PAP_full','A','T','O'],['AAC resemblance','ShortStop (n=1,057)','PAP C30','PAP full (secondary)','A: composition','T: + terminals','O: + native order']);panel(ax[0],'A','WT reporter ranking (n=1,072)')
dots(ax[1],'additional_strict',['AAC_resemblance','ShortStop','PAP_Cterminal','A','T','O','Guo_regression'],['AAC resemblance','ShortStop (n=595)','PAP C30','A: composition','T: + terminals','O: + native order','Guo released regression']);panel(ax[1],'B','Additional strict transfer (n=602)')
fig.tight_layout(w_pad=2);save(fig,'Figure2')
c=json.loads((R/'native_contrasts.json').read_text());z=pd.DataFrame(c);names=['Primary GFP-MP','MP-mCherry','Mutagenesis WT','40% identity grouping','Fold seed +1','Fold seed +2'];fig,ax=plt.subplots(1,2,figsize=(9,4.3))
for a,key,title in [(ax[0],'rank_gain','Order-minus-terminal rank gain'),(ax[1],'mae_reduction','Terminal-minus-order MAE')]:
    x=z[key].to_numpy();bounds=np.asarray(z[key+'_CI'].tolist());a.errorbar(x,np.arange(len(x)),xerr=[x-bounds[:,0],bounds[:,1]-x],fmt='o',c=blue,capsize=3);a.axvline(0,c=grey,lw=.8);a.set_yticks(np.arange(len(x)),names);a.invert_yaxis();a.set_xlabel(title);a.grid(axis='x',alpha=.15)
panel(ax[0],'A','Incremental native order');panel(ax[1],'B','Error reduction');fig.tight_layout();save(fig,'Figure3');z.to_json(D/'Figure3_source.json',orient='records',indent=2)
fig,ax=plt.subplots(1,2,figsize=(9.8,4.5));models=['family_mean','T','O','PAP_Cterminal_calibrated','PAP_full_calibrated','delta_trained_T','cross_family_T'];labels=['Family mean','WT-trained T','WT-trained O','Calibrated PAP C30','Calibrated PAP full','Delta-trained T','Opposite-family T']
rows=[]
for a,fam,title in [(ax[0],'L_to_A','Leu-to-Ala (n=1,014)'),(ax[1],'microprotein_10mer_ext','C-terminal extension (n=1,007)')]:
    vals=[metric('mutation_'+fam,mod,field='mae') for mod in models];a.barh(np.arange(len(models)),vals,color=[grey,blue,blue,green,green,orange,orange]);a.set_yticks(np.arange(len(models)),labels);a.invert_yaxis();a.set_xlim(0,.19);a.set_xlabel('MAE of predicted reporter-score change');panel(a,'A' if fam=='L_to_A' else 'B',title)
    for i,(mod,val) in enumerate(zip(models,vals)):rows.append(dict(family=fam,model=mod,mae=val));a.text(val+.002,i,f'{val:.3f}',va='center',fontsize=8)
fig.tight_layout(w_pad=2);save(fig,'Figure4');pd.DataFrame(rows).to_csv(D/'Figure4_source.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(10,4.3),gridspec_kw={'width_ratios':[1.4,1]})
dots(ax[0],'Yang_strict',['published_PepScore','Length','PAP_Cterminal_fusion_sequence','PAP_full_fusion_sequence','MP_mCherry_T_fusion_sequence','MP_mCherry_O_fusion_sequence','GFP_MP_T_native_sequence'],['Published PepScore','Native length','PAP C30 + FLAG','PAP full + FLAG','mCherry-trained T + FLAG','mCherry-trained O + FLAG','GFP-trained T, native']);panel(ax[0],'A','Independent study (strict n=20)')
mask=v.strict_disjoint;ax[1].scatter(v.loc[~mask,'native_length'],v.loc[~mask,'untreated_mean'],s=30,facecolors='none',edgecolors=grey,label='Other 17');ax[1].scatter(v.loc[mask,'native_length'],v.loc[mask,'untreated_mean'],s=32,c=blue,label='Strict 20');ax[1].axhline(.05,c=grey,lw=.8,ls='--');ax[1].set_xlabel('Native peptide length (aa)');ax[1].set_ylabel('Normalized FLAG-positive cell fraction');ax[1].legend(frameon=False,fontsize=8);panel(ax[1],'B','Assay and selection context');fig.tight_layout(w_pad=2);save(fig,'Figure5');v[['orf_name','native_length','untreated_mean','strict_disjoint']].to_csv(D/'Figure5_source.csv',index=False)
# Historical controlled clustering context, with unchanged source and verification.
legacy=P/'supporting/FigureS1_source.png'
if not legacy.exists():legacy=P.parent/'thesis_reanalysis_2026-10-31/figures/Figure4.png'
shutil.copy2(legacy,D/'FigureS1.png')
print('Five benchmark figures and controlled clustering context written.')
