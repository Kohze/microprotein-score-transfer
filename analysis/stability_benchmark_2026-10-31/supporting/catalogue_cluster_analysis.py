"""Original-space and PCA clustering with seed, sampling, family and null checks."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='clusters');args=parser.parse_args()
    out=ROOT/args.output_dir;out.mkdir(exist_ok=True)
    d=pd.read_parquet(ROOT/'matched_comprehensive.parquet')
    ids=pd.read_csv(ROOT/'features_native/row_ids.csv').id;idx={x:i for i,x in enumerate(ids)}
    x=np.load(ROOT/'features_native/matrix.npy',mmap_mode='r')[[idx[i] for i in d.id]]
    cids=pd.read_csv(ROOT/'features_controls/row_ids.csv').id;cidx={x:i for i,x in enumerate(cids)}
    sx=np.load(ROOT/'features_controls/matrix.npy',mmap_mode='r')[[cidx[i+'__shuffle0'] for i in d.id]]
    spec=json.loads((ROOT/'features_native/feature_definition.json').read_text());families=np.array(spec['families'])
    assert np.isfinite(x).all() and np.isfinite(sx).all()
    y=d['class'].eq('nORF').to_numpy(int)
    scaler=StandardScaler().fit(x);z=scaler.transform(x);sz=scaler.transform(sx)
    counts={f:int((families==f).sum()) for f in set(families)}
    weights=np.array([1/np.sqrt(counts[f]) for f in families])
    results=[];labels={};coords={};original_labels={}
    with threadpool_limits(limits=4):
        for scheme,scale in [('per_feature',np.ones(x.shape[1])),('per_family',weights)]:
            zz=z*scale;ss=sz*scale
            pca=PCA(n_components=50,svd_solver='randomized',random_state=20261031).fit(zz)
            pc=pca.transform(zz);spc=pca.transform(ss)
            coords[scheme]=pc
            np.savez_compressed(out/f'{scheme}_pca.npz',native=pc,shuffle=spc,variance=pca.explained_variance_ratio_)
            for k in [2,3,5,8]:
                native_labels=[]
                for seed in [10,20,30]:
                    model=KMeans(k,n_init=10,max_iter=300,random_state=seed).fit(pc)
                    lab=model.labels_;native_labels.append(lab)
                    shuffle_model=KMeans(k,n_init=10,max_iter=300,random_state=seed).fit(spc)
                    rng=np.random.default_rng(seed);sub=rng.choice(len(d),int(.8*len(d)),replace=False)
                    sub_model=KMeans(k,n_init=10,max_iter=300,random_state=seed).fit(pc[sub])
                    results.append(dict(scheme=scheme,space='PCA50',k=k,seed=seed,
                        silhouette_native=float(silhouette_score(pc,lab,sample_size=min(1000,len(d)),random_state=42)),
                        silhouette_shuffle=float(silhouette_score(spc,shuffle_model.labels_,sample_size=min(1000,len(d)),random_state=42)),
                        label_ARI=float(adjusted_rand_score(y,lab)),
                        native_shuffle_ARI=float(adjusted_rand_score(lab,shuffle_model.labels_)),
                        subsample_ARI=float(adjusted_rand_score(lab,sub_model.predict(pc))),
                        class_silhouette=float(silhouette_score(pc,y,sample_size=min(1000,len(d)),random_state=42)),
                        PCA50_variance=float(pca.explained_variance_ratio_.sum())))
                    labels[(scheme,k,seed)]=lab
                for r in results[-3:]:r['seed_ARI']=float(np.mean([adjusted_rand_score(native_labels[i],native_labels[j]) for i in range(3) for j in range(i)]))
            for k in [2,5]:
                model=MiniBatchKMeans(k,n_init=5,batch_size=512,max_iter=200,random_state=42).fit(zz)
                smodel=MiniBatchKMeans(k,n_init=5,batch_size=512,max_iter=200,random_state=42).fit(ss)
                original_labels[(scheme,k)]=model.labels_
                results.append(dict(scheme=scheme,space='original1440',k=k,seed=42,
                    silhouette_native=float(silhouette_score(zz,model.labels_,sample_size=min(1000,len(d)),random_state=42)),
                    silhouette_shuffle=float(silhouette_score(ss,smodel.labels_,sample_size=min(1000,len(d)),random_state=42)),
                    label_ARI=float(adjusted_rand_score(y,model.labels_)),
                    native_shuffle_ARI=float(adjusted_rand_score(model.labels_,smodel.labels_))))
        for k in [2,3,5,8]:
            value=float(adjusted_rand_score(labels[('per_feature',k,10)],labels[('per_family',k,10)]))
            for r in results:
                if r['k']==k and r['space']=='PCA50':
                    r['family_weighting_ARI']=value
                    r['family_weighting_reference_seed']=10
                elif r['k']==k and r['space']=='original1440':
                    r['family_weighting_ARI']=float(adjusted_rand_score(original_labels[('per_feature',k)],original_labels[('per_family',k)]))
                    r['family_weighting_reference_seed']=42
        # t-SNE coordinates are descriptive only; assess no hypothesis in 2-D.
        for perplexity,seed in [(30,10),(50,20)]:
            tsne=TSNE(n_components=2,perplexity=perplexity,init='pca',learning_rate='auto',
                n_iter=1000,random_state=seed,n_jobs=4).fit_transform(coords['per_family'])
            pd.DataFrame({'id':d.id,'class':d['class'],'x':tsne[:,0],'y':tsne[:,1]}).to_csv(out/f'tsne_p{perplexity}_s{seed}.csv',index=False)
    pd.DataFrame(results).to_csv(out/'cluster_metrics.csv',index=False)
    np.savez_compressed(out/'cluster_labels.npz',**{f'{s}_k{k}_s{seed}':lab for (s,k,seed),lab in labels.items()})
    np.savez_compressed(out/'original_space_cluster_labels.npz',**{f'{s}_k{k}_s42':lab for (s,k),lab in original_labels.items()})
    d[['id','class','length','pair']].to_csv(out/'cluster_row_metadata.csv',index=False)
    (out/'completion.json').write_text(json.dumps(dict(rows=len(d),metrics=len(results),functional_class_validation=False),indent=2))
    print(pd.DataFrame(results).groupby(['scheme','space','k'])[['silhouette_native','silhouette_shuffle','label_ARI','native_shuffle_ARI']].mean().round(4).to_string())

if __name__=='__main__':main()
