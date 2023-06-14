from sklearn.preprocessing import MinMaxScaler, Normalizer
import scanpy as sc
import numpy as np

def scale(
    adata,
    batch_correction=True,
    normalize=True,
    minmax_scale=True,
    log=True,
    target_sum=1e3,
    keep_raw=True,
    normalize_mode='l2' #sklearn Normalizer norm parameters
    ):

    if keep_raw:
        adata.raw = adata

    if normalize:
        if normalize_mode == 'sum':
            sc.pp.normalize_total(adata, target_sum=target_sum)
        else:
            adata.X = Normalizer(norm=normalize_mode).fit_transform(adata.X)
    if log:
        sc.pp.log1p(adata)
    
    if batch_correction:    
        adatas = []
        for s in np.unique(adata.obs.Sample):
            ad = adata[adata.obs.Sample == s,]#.X.mean(axis=0)
            ad.X = MinMaxScaler().fit_transform(ad.X)
            adatas.append(ad)
        adata = adata[0].concatenate(*adatas[1:])
    else:
        adata.X = MinMaxScaler().fit_transform(adata.X)
