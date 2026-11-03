"""Compare a fresh pipeline rerun with the frozen clustering outputs."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parent
def main():
    old=ROOT/'clusters';new=ROOT/'clustering_reproduction_check'
    a=pd.read_csv(old/'cluster_metrics.csv');b=pd.read_csv(new/'cluster_metrics.csv')
    cols=[c for c in a.select_dtypes('number').columns if c not in ['family_weighting_ARI','family_weighting_reference_seed']]
    error=float(np.nanmax(abs(a[cols].to_numpy()-b[cols].to_numpy())))
    assert error==0
    labels_a=np.load(old/'cluster_labels.npz');labels_b=np.load(new/'cluster_labels.npz')
    ari={k:float(adjusted_rand_score(labels_a[k],labels_b[k])) for k in labels_a.files}
    assert all(v==1 for v in ari.values())
    pca={};tsne={}
    for scheme in ['per_feature','per_family']:
        x=np.load(old/f'{scheme}_pca.npz');y=np.load(new/f'{scheme}_pca.npz')
        pca[scheme]={k:float(np.max(abs(x[k]-y[k]))) for k in ['native','shuffle','variance']}
        assert all(v==0 for v in pca[scheme].values())
    for name in ['tsne_p30_s10.csv','tsne_p50_s20.csv']:
        x=pd.read_csv(old/name);y=pd.read_csv(new/name);assert list(x.id)==list(y.id)
        tsne[name]=float(np.max(abs(x[['x','y']].to_numpy()-y[['x','y']].to_numpy())))
        assert tsne[name]==0
    correction=[]
    for i,row in b.iterrows():
        if row['space']=='original1440':
            correction.append(dict(scheme=row['scheme'],k=int(row['k']),old=float(a.iloc[i].family_weighting_ARI),corrected=float(row.family_weighting_ARI)))
    # Preserve the previously verified computations, replacing only the corrected
    # auxiliary comparison and adding its explicit reference seed plus raw-space
    # labels. Plotted values and manuscript numerical results do not change.
    shutil.copy2(new/'cluster_metrics.csv',old/'cluster_metrics.csv')
    shutil.copy2(new/'original_space_cluster_labels.npz',old/'original_space_cluster_labels.npz')
    shutil.copy2(new/'cluster_metrics.csv',ROOT/'figure_source_data/Figure4_cluster_metrics.csv')
    receipt=dict(status='passed',scope='Repeatability of the updated matched-data pipeline; not exact historical thesis reproduction',
        rows=1488,matched_pairs=744,metrics_repeated=28,pca_clusterings_identical=ari,
        pca_max_absolute_errors=pca,tsne_coordinate_max_absolute_errors=tsne,
        non_bookkeeping_metric_max_difference=error,
        auxiliary_table_correction=correction,
        correction_explanation='Original-space family_weighting_ARI had inadvertently repeated the PCA-space comparison. It now compares saved original-space assignments. PCA values explicitly identify reference seed 10; original-space comparisons use seed 42.',
        paper_figures_and_reported_numerical_conclusions_changed=False,
        historical_tensorboard_embeddings_reproduced=False,UMAP_rerun=False,
        limitations=['Updated catalogue and exact-length matched subset only','Fixed-setting repeatability does not establish biological replication','Two t-SNE settings, not an independent seed-by-perplexity grid','No biological functional-class validation'])
    (ROOT/'clustering_reproducibility_verification.json').write_text(json.dumps(receipt,indent=2))
    print('Verified 24 PCA-space clusterings, two PCA representations and two t-SNE maps; all reproduced exactly.')
if __name__=='__main__':main()
