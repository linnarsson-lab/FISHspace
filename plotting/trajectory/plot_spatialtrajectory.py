from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm


def plot_spatialtrajectory(
    adata,
    plot_method:str = 'stream',
    scale:str = 1.0,
    grid_density:str = 1,
    grid_knn:str = None,
    grid_scale:float = 1,
    grid_thresh:float = 1.0,
    grid_width:float = 0.005,
    arrow_color:str = 'black',
    stream_cutoff_perc:float = 5,
    stream_linewidth:float = 1,
    stream_density:float = 1.0,
    ax = None,
    filename=None,
    ):

    X = adata.uns['vector_field_origin']
    V = adata.uns['vector_field_delta']
    ncell = X.shape[0]

    V_cell = V.copy()
    V_cell_sum = np.sum(V_cell, axis=1)
    V_cell[np.where(V_cell_sum==0)[0],:] = np.nan

    if ax == None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    if plot_method == "grid" or plot_method == "stream":
        # Get a rectangular grid
        xl, xr = np.min(X[:,0]), np.max(X[:,0])
        epsilon = 0.02*(xr-xl); xl -= epsilon; xr += epsilon
        yl, yr = np.min(X[:,1]), np.max(X[:,1])
        epsilon = 0.02*(yr-yl); yl -= epsilon; yr += epsilon
        ngrid_x = int(50 * grid_density)
        gridsize = (xr-xl) / float(ngrid_x)
        ngrid_y = int((yr-yl)/gridsize)
        meshgrid = np.meshgrid(np.linspace(xl,xr,ngrid_x), np.linspace(yl,yr,ngrid_y))
        grid_pts = np.concatenate((meshgrid[0].reshape(-1,1), meshgrid[1].reshape(-1,1)), axis=1)

        if grid_knn is None:
            grid_knn = int( X.shape[0] / 50 )
        nn_mdl = NearestNeighbors()
        nn_mdl.fit(X)
        dis, nbs = nn_mdl.kneighbors(grid_pts, n_neighbors=grid_knn)
        w = norm.pdf(x=dis, scale=gridsize * grid_scale)
        w_sum = w.sum(axis=1)

        V_grid = (V[nbs] * w[:,:,None]).sum(axis=1)
        V_grid /= np.maximum(1, w_sum)[:,None]

        if plot_method == "grid":
            grid_thresh *= np.percentile(w_sum, 99) / 100
            grid_pts, V_grid = grid_pts[w_sum > grid_thresh], V_grid[w_sum > grid_thresh]
        elif plot_method == "stream":
            x_grid = np.linspace(xl, xr, ngrid_x)
            y_grid = np.linspace(yl, yr, ngrid_y)
            V_grid = V_grid.T.reshape(2, ngrid_y, ngrid_x)
            vlen = np.sqrt((V_grid ** 2).sum(0))
            grid_thresh = 10 ** (grid_thresh - 6)
            grid_thresh = np.clip(grid_thresh, None, np.max(vlen) * 0.9)
            cutoff = vlen.reshape(V_grid[0].shape) < grid_thresh
            length = np.sum(np.mean(np.abs(V[nbs]),axis=1),axis=1).T
            length = length.reshape(ngrid_y, ngrid_x)
            cutoff |= length < np.percentile(length, stream_cutoff_perc)

            V_grid[0][cutoff] = np.nan


    if plot_method == "cell":
        ax.quiver(X[:,0], X[:,1], V_cell[:,0], V_cell[:,1], scale=scale, scale_units='x', color=arrow_color)
    elif plot_method == "grid":
        ax.quiver(grid_pts[:,0], grid_pts[:,1], V_grid[:,0], V_grid[:,1], scale=scale, scale_units='x', width=grid_width, color=arrow_color)
    elif plot_method == "stream":
        lengths = np.sqrt((V_grid ** 2).sum(0))
        stream_linewidth *= 2 * lengths / lengths[~np.isnan(lengths)].max()
        ax.streamplot(x_grid, y_grid, V_grid[0], V_grid[1], color=arrow_color, density=stream_density, linewidth=stream_linewidth)

    ax.axis("equal")
    ax.axis("off")
    if not filename is None:
       plt.savefig(filename, dpi=500, bbox_inches = 'tight', transparent=True)