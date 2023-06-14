import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import KDTree
from tqdm import tqdm, trange
import logging
from sklearn.manifold import SpectralEmbedding
from itertools import permutations, product
from collections import Counter
logging.basicConfig(level=logging.INFO)

def umap(
        adata,
        n_obs: int = 250000, # Number of samples to run leiden on
        n_neighbors: int = 15, # Number of neighbors to use for leiden
        n_epochs: int = 250, # Number of epochs to run UMAP
        ):

    from umap.parametric_umap import ParametricUMAP
    adata_mini = sc.pp.subsample(adata, n_obs=n_obs, copy=True)
    embedder = ParametricUMAP(n_epochs = n_epochs, verbose=True)
    embedding_mini = embedder.fit_transform(adata_mini.obsm['X_pca'])

    history = embedder._history
    embedding = embedder.fit_transform(adata.obsm['X_pca'])
    adata.obsm['X_umap_parametric'] = embedding
