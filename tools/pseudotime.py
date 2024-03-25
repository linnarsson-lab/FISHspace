import scanpy as sc
import numpy as np
import pandas as pd
import cellrank as cr
import logging
from FISHspace.tools.utils.cellrank_utils import _correlation_test
from FISHspace.pp import preprocess
from cellrank.pl._utils import (
    _fit_bulk,
    _get_backend,
    _create_models,
    _create_callbacks,
)

from cellrank.ul._utils import (
    _get_n_cores
)
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from typing import Union, Optional, Iterable

import numpy as np
import pandas as pd
from functools import partial
from anndata import AnnData
import shutil
import sys
import copy
from statsmodels.stats.multitest import multipletests
import igraph
import warnings
from numba import cuda, njit, prange
import math
from tqdm import tqdm
from joblib import Parallel
from scipy import sparse
import numpy as np
import shutil
from joblib import delayed
from tqdm import tqdm


def importeR(task, module="mgcv"):
    try:
        from rpy2.robjects import pandas2ri, Formula
        from rpy2.robjects.packages import PackageNotInstalledError, importr
        import rpy2.rinterface

        pandas2ri.activate()
        Rpy2 = True
    except ModuleNotFoundError as e:
        Rpy2 = (
            "rpy2 installation is necessary for "
            + task
            + '. \
            \nPlease use "pip3 install rpy2" to install rpy2'
        )
        Formula = False

    if not shutil.which("R"):
        R = (
            "R installation is necessary for "
            + task
            + ". \
            \nPlease install R and try again"
        )
    else:
        R = True

    try:
        rstats = importr("stats")
    except Exception as e:
        rstats = (
            "R installation is necessary for "
            + task
            + ". \
            \nPlease install R and try again"
        )

    try:
        rmodule = importr(module)
    except Exception as e:
        rmodule = (
            f'R package "{module}" is necessary for {task}'
            + "\nPlease install it and try again"
        )

    return Rpy2, R, rstats, rmodule, Formula

def get_X(adata, cells, genes, layer, togenelist=False):
    if layer is None:
        if sparse.issparse(adata.X):
            X = adata[cells, genes].X.A
        else:
            X = adata[cells, genes].X
    else:
        if sparse.issparse(adata.layers[layer]):
            X = adata[cells, genes].layers[layer].A
        else:
            X = adata[cells, genes].layers[layer]

    if togenelist:
        return X.T.tolist()
    else:
        return X
    
def getpath(g, root, tips, tip, tree, df):
    import warnings
    import numpy as np

    wf = warnings.filters.copy()
    warnings.filterwarnings("ignore")
    try:
        path = np.array(g.vs[:]["name"])[
            np.array(g.get_shortest_paths(str(root), str(tip)))
        ][0]
        segs = list()
        for i in range(len(path) - 1):
            segs = segs + [
                np.argwhere(
                    (
                        tree["pp_seg"][["from", "to"]]
                        .astype(str)
                        .apply(lambda x: all(x.values == path[[i, i + 1]]), axis=1)
                    ).to_numpy()
                )[0][0]
            ]
        segs = tree["pp_seg"].index[segs]
        pth = df.loc[df.seg.astype(int).isin(segs), :].copy(deep=True)
        pth["branch"] = str(root) + "_" + str(tip)
        warnings.filters = wf
        return pth
    except IndexError:
        pass


class ProgressParallel(Parallel):
    def __init__(
        self, use_tqdm=True, total=None, file=None, desc=None, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._file = file
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(
            disable=not self._use_tqdm,
            total=self._total,
            desc=self._desc,
            file=self._file,
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

Rpy2, R, rstats, rmgcv, Formula = importeR("testing feature association to the tree")
check = [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]


def pseudotime_genes(
    adata,
    genes_list,
    cluster_key,
    clusters=None,
    latent_time_key = 'latent_time',
    n_bins=4,
    min_pval=0.005,
    min_logfold=0.1,
    highlight_genes=[],

    ):
    adata.obs['latent_time']
    adata_orig = adata.copy()

    if clusters is not None:
        b00l = np.isin(adata.obs[cluster_key], clusters)
        adata = adata[b00l,:]
    #sc.pp.normalize_per_cell(adata)
    #sc.pp.log1p(adata)

    model = cr.ul.models.GAM(adata)

    lineage_class = cr.tl.Lineage(
                                    input_array=adata.obs['latent_time'].values,
                                    names=['latent'],
                                )
    adata.obsm['to_terminal_states'] = lineage_class
    driver_adata = _correlation_test(
                                    adata.X,
                                    Y=lineage_class,
                                    gene_names=adata.var.index,
                                    )

    notnull=~(driver_adata['latent_corr'].isnull())
    driver_adata = driver_adata.loc[notnull,:]
    idx = np.argsort(driver_adata['latent_corr'])[::-1]
    top_cell = driver_adata.iloc[idx,:]

    b00l = np.isin(top_cell.index.values,genes_list)
    top_lineage_genes = top_cell.loc[b00l,:]
    b00l = np.isin(adata.obs[cluster_key],clusters)

    time_adata = adata[b00l,:] 
    cellid_bin_ =[]

    if type(n_bins) == int:
        edges = [int((x)*(time_adata.shape[0]/n_bins )) for x in range(n_bins+1)]

    elif type(n_bins) == list:
        n_bins_list = n_bins
        n_bins = len(n_bins) - 1
        edges = [int((x)*(time_adata.shape[0]/n_bins )) for x in n_bins_list]

    for n in range(n_bins):
        cellid_bin = time_adata.obs.iloc[np.argsort(time_adata.obs['latent_time'].values)][edges[n]:edges[n+1]].index
        cellid_bin_.append(cellid_bin)

    time_adata.obs['lineage_Clusters']=np.repeat(0,time_adata.shape[0])
    for i,v in enumerate(cellid_bin_):
        b00l = np.isin(time_adata.obs.index,v)
        time_adata.obs['lineage_Clusters'][b00l] =np.repeat(i,len(b00l))
    time_adata.obs['lineage_Clusters'] = time_adata.obs['lineage_Clusters'].astype('category')
    
    
    if np.array_equal(adata.obs_names, time_adata.obs_names):
        adata.obs['lineage_Clusters'] = time_adata.obs['lineage_Clusters'].values
        b00l_var_1 = np.isin(adata.var.index,top_lineage_genes.index)
        sub_adata = adata[:,b00l_var_1]
        #sc.pp.normalize_per_cell(sub_adata)
        #
        #sc.pp.log1p(sub_adata)
        #del sub_adata.raw

        sc.tl.rank_genes_groups(sub_adata, 'lineage_Clusters', use_raw=False)

        result = sub_adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        rank_result = pd.DataFrame(
            {group + '_' + key: result[key][group]
            for group in groups for key in ['names', 'pvals_adj','logfoldchanges']})
    else:
        print(f'CELL ID NOT the SAME!')

    bins=n_bins
    gene_in_order = []
    for i in np.arange(bins):

        total_genes= 1000
        min_logfold_ = min_logfold
        counter = 0
        print('bin',i,)
        while total_genes > 100:
            #print('bin',i,)
            b00l_0 = rank_result[f'{i}_pvals_adj']<min_pval
            #print(b00l_0.sum(),'pval')

            b00l_1 = rank_result[f'{i}_logfoldchanges']>min_logfold_
            bool_names = rank_result[f'{i}_names'].isin(highlight_genes) & rank_result[f'{i}_logfoldchanges']>min_logfold
            b00l = b00l_1 | bool_names
            #print(b00l_1.sum(), 'lofgold')
            b00l = b00l_0 & b00l_1
            #print(b00l.sum(),'left genes')
            rank_result_thres = rank_result.loc[b00l,:]
            min_logfold_ += 0.2
            total_genes = rank_result_thres.shape[0]
            counter += 1
            if counter > 100:  
                break
        if rank_result_thres.shape[0]>100:
            if i == (bins-1):
                print(i)
                print(rank_result_thres)
                print(rank_result_thres[f'{i}_logfoldchanges'])
                thres = np.percentile(rank_result_thres[f'{i}_logfoldchanges'].values,50)
                b00l = rank_result_thres[f'{i}_logfoldchanges'].values > thres
                bool_names = rank_result_thres[f'{i}_names'].isin(highlight_genes)
                b00l = b00l | bool_names

                rank_result_thres = rank_result_thres.loc[b00l,:]
        else:
            if i == (bins-1):
                thres = np.percentile(rank_result_thres[f'{i}_logfoldchanges'].values,50)
                b00l = rank_result_thres[f'{i}_logfoldchanges'].values>thres
                bool_names = rank_result_thres[f'{i}_names'].isin(highlight_genes)
                b00l = b00l | bool_names

                rank_result_thres = rank_result_thres.loc[b00l,:]


        sort_ = np.argsort(rank_result_thres[f'{i}_logfoldchanges'].values)[::-1]
        result_sorted = rank_result_thres.iloc[sort_]
        #print(result_sorted.shape[0],'result_sorted')
        gene_in_order.append(result_sorted[f'{i}_names'].values)
        
        tidy_gene_in_order =[]
        for i, v in enumerate(gene_in_order):
            flat = [i for s in tidy_gene_in_order for i in s]
            b00l = np.isin(v,flat,invert=True)
            tidy_gene_in_order.append(v[b00l])
        tidy_gene_in_order = [i for s in tidy_gene_in_order for i in s]

    try:
        adata_orig.X = adata_orig.raw[:,adata_orig.var_names].X
    except:
        pass
    data_processed = _preprocess(
        model,
        top_lineage_genes,
        sub_adata,

    )
    lineages_out = dict(
            {
                'tidy_gene_in_order':tidy_gene_in_order,
                'time_adata':time_adata,
                'model':model,
                'top_lineage_genes':top_lineage_genes,
                'data_processed':data_processed,
                'sub_adata':sub_adata,
            }
        )
    
        
    return lineages_out

def _preprocess(
    model,
    top_lineage_genes,
    adata,
    ):
    lineages = ['latent']
    
    orig_ = adata

    models = _create_models(model, top_lineage_genes.index,lineages)
    callback = None
    time_range = None
    backend = "loky"
    n_jobs = 1
    show_progress_bar = True
    kwargs = dict()
    kwargs["backward"] = False
    kwargs["time_key"] = 'latent_time'
    # kwargs['n_test_points']=None
    all_models, data, genes, lineages = _fit_bulk(
        models,
        _create_callbacks(orig_, callback, top_lineage_genes.index, lineages, **kwargs),
        top_lineage_genes.index,
        lineages,
        time_range,
        return_models=True,  # always return (better error messages)
        filter_all_failed=True,
        parallel_kwargs={
            "show_progress_bar": show_progress_bar,
            "n_jobs": _get_n_cores(n_jobs, len(top_lineage_genes.index)),
            "backend": _get_backend(models, backend),
        },
        **kwargs,
    )
    return data


def test_association(
    adata: AnnData,
    n_map: int = 1,
    n_jobs: int = 1,
    spline_df: int = 5,
    fdr_cut: float = 1e-4,
    A_cut: int = 1,
    st_cut: float = 0.8,
    reapply_filters: bool = False,
    plot: bool = False,
    copy: bool = False,
    layer: Optional[str] = None,
):

    """\
    Determine a set of genes significantly associated with the trajectory.


    Feature expression is modeled as a function of pseudotime in a branch-specific manner,
    using cubic spline regression :math:`g_{i} \\sim\ t_{i}` for each branch independently.
    This tree-dependent model is then compared with an unconstrained model :math:`g_{i} \\sim\ 1`
    using F-test.

    The models are fit using *mgcv* R package.

    Benjamini-Hochberg correction is used to adjust for multiple hypothesis testing.


    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        adata layer to use for the test.
    n_map
        number of cell mappings from which to do the test.
    n_jobs
        number of cpu processes used to perform the test.
    spline_df
        dimension of the basis used to represent the smooth term.
    fdr_cut
        FDR (Benjamini-Hochberg adjustment) cutoff on significance; significance if FDR < fdr_cut.
    A_cut
        amplitude is max of predicted value minus min of predicted value by GAM. significance if A > A_cut.
    st_cut
        cutoff on stability (fraction of mappings with significant (fdr,A) pair) of association; significance, significance if st > st_cut.
    reapply_filters
        avoid recomputation and reapply fitlers.
    plot
        call scf.pl.test_association after the test.
    root
        restrain the test to a subset of the tree (in combination with leaves).
    leaves
        restrain the test to a subset of the tree (in combination with root).
    copy
        Return a copy instead of writing to adata.
    Returns
    -------
    adata : anndata.AnnData
        if `copy=True` it returns or else add fields to `adata`:

        `.var['p_val']`
            p-values from statistical test.
        `.var['fdr']`
            corrected values from multiple testing.
        `.var['st']`
            proportion of mapping in which feature is significant.
        `.var['A']`
            amplitue of change of tested feature.
        '.var['signi']`
            feature is significantly changing along pseuodtime
        `.uns['stat_assoc_list']`
            list of fitted features on the tree for all mappings.
    """

    if any(check):
        idx = np.argwhere(
            [type(imp) == str for imp in [Rpy2, R, rstats, rmgcv, Formula]]
        ).min()
        raise Exception(np.array([Rpy2, R, rstats, rmgcv, Formula])[idx])

    adata = adata.copy() if copy else adata

    if "t" not in adata.obs:
        raise ValueError(
            "You need to run `tl.pseudotime` before testing for association."
            + "Or add a precomputed pseudotime at adata.obs['t'] for single segment."
        )

    if reapply_filters & ("stat_assoc_list" in adata.uns):
        stat_assoc_l = list(adata.uns["stat_assoc_list"].values())
        # stat_assoc_l = list(map(lambda x: pd.DataFrame(x,index=x["features"]),stat_assoc_l))
        adata = apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut)

        logging.info(
            "reapplied filters, "
            + str(sum(adata.var["signi"]))
            + " significant features"
        )

        if plot:
            plot_test_association(adata)

        return adata if copy else None

    Xgenes = get_X(adata, adata.obs_names, adata.var_names, layer, togenelist=True)

    logging.info("test features for association with the trajectory")

    stat_assoc_l = list()

    def test_assoc_map(m):
        if n_map == 1:
            df = adata.obs.loc[:, ["t", "seg"]]
        else:
            df = adata.uns["pseudotime_list"][str(m)]
        data = list(zip([df] * len(Xgenes), Xgenes))

        stat = ProgressParallel(
            n_jobs=n_jobs if n_map == 1 else 1,
            total=len(data),
            file=sys.stdout,
            use_tqdm=n_map == 1,
            desc="    single mapping ",
        )(delayed(test_assoc)(data[d], spline_df) for d in range(len(data)))
        stat = pd.DataFrame(stat, index=adata.var_names, columns=["p_val", "A"])
        stat["fdr"] = multipletests(stat.p_val, method="bonferroni")[1]
        return stat

    stat_assoc_l = ProgressParallel(
        n_jobs=1 if n_map == 1 else n_jobs,
        total=n_map,
        file=sys.stdout,
        use_tqdm=n_map > 1,
        desc="    multi mapping ",
    )(delayed(test_assoc_map)(m) for m in range(n_map))

    adata = apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut)

    logging.info(
        "    found " + str(sum(adata.var["signi"])) + " significant features",
    )
    logging.info(
        "added\n"
        "    .var['p_val'] values from statistical test.\n"
        "    .var['fdr'] corrected values from multiple testing.\n"
        "    .var['st'] proportion of mapping in which feature is significant.\n"
        "    .var['A'] amplitue of change of tested feature.\n"
        "    .var['signi'] feature is significantly changing along pseudotime.\n"
        "    .uns['stat_assoc_list'] list of fitted features on the graph for all mappings."
    )

    if plot:
        plot_test_association(adata)

    return adata if copy else None

def test_assoc(data, spline_df):
    sdf = data[0]
    sdf["exp"] = data[1]

    global rmgcv
    global rstats

    def gamfit(s):
        m = rmgcv.gam(
            Formula(f"exp~s(t,k={spline_df})"), data=sdf.loc[sdf["seg"] == s, :]
        )
        return dict({"d": m[5][0], "df": m[42][0], "p": rmgcv.predict_gam(m)})

    mdl = list(map(gamfit, sdf.seg.unique()))
    mdf = pd.concat(list(map(lambda x: pd.DataFrame([x["d"], x["df"]]), mdl)), axis=1).T
    mdf.columns = ["d", "df"]

    odf = sum(mdf["df"]) - mdf.shape[0]
    m0 = rmgcv.gam(Formula("exp~1"), data=sdf)
    if sum(mdf["d"]) == 0:
        fstat = 0
    else:
        fstat = (m0[5][0] - sum(mdf["d"])) / (m0[42][0] - odf) / (sum(mdf["d"]) / odf)

    df_res0 = m0[42][0]
    df_res_odf = df_res0 - odf
    pval = rstats.pf(fstat, df_res_odf, odf, lower_tail=False)[0]
    pr = np.concatenate(list(map(lambda x: x["p"], mdl)))

    return [pval, max(pr) - min(pr)]


def apply_filters(adata, stat_assoc_l, fdr_cut, A_cut, st_cut, prefix=""):
    n_map = len(stat_assoc_l)
    if n_map > 1:
        stat_assoc = pd.DataFrame(
            {
                prefix
                + "p_val": pd.concat(
                    list(map(lambda x: x[prefix + "p_val"], stat_assoc_l)), axis=1
                ).median(axis=1),
                prefix
                + "A": pd.concat(
                    list(map(lambda x: x[prefix + "A"], stat_assoc_l)), axis=1
                ).median(axis=1),
                prefix
                + "fdr": pd.concat(
                    list(map(lambda x: x[prefix + "fdr"], stat_assoc_l)), axis=1
                ).median(axis=1),
                prefix
                + "st": pd.concat(
                    list(
                        map(
                            lambda x: (x[prefix + "fdr"] < fdr_cut)
                            & (x[prefix + "A"] > A_cut),
                            stat_assoc_l,
                        )
                    ),
                    axis=1,
                ).sum(axis=1)
                / n_map,
            }
        )
    else:
        stat_assoc = stat_assoc_l[0]
        stat_assoc[prefix + "st"] = (
            (stat_assoc[prefix + "fdr"] < fdr_cut) & (stat_assoc[prefix + "A"] > A_cut)
        ) * 1

    # saving results
    stat_assoc[prefix + "signi"] = stat_assoc[prefix + "st"] > st_cut
    for c in stat_assoc.columns:
        adata.var[c] = stat_assoc[c]

    names = np.arange(len(stat_assoc_l)).astype(str).tolist()

    dictionary = dict(zip(names, stat_assoc_l))
    adata.uns[prefix + "stat_assoc_list"] = dictionary

    return adata