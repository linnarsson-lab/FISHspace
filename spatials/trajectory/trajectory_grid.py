import matplotlib.pyplot as plt
import scanpy as sc
from typing import Optional, Dict, Any, Callable, Tuple
import math
from scipy.spatial import KDTree
import numpy as np
import pandas as pd
from collections import Counter
import squidpy as sq
from sklearn.preprocessing import normalize


def vector_field(
    adata,
    sample:str,
    cluster_key:str = 'CombinedNameMerge',
    clusters:list = None,
    k:int=6, #
    spacing:int=100,
    min_count:int=2,
    remove_negative_values=False,
    direction:int = 1, # 1:outward, -1:inward
    copy:bool=False,
    ):

    adata_s = adata[adata.obs.Sample == sample]

    if clusters is not None:
        adata_s = adata_s[adata_s.obs[cluster_key].isin(clusters)]

    sq.gr.spatial_neighbors(adata_s, coord_type="generic", n_neighs=10,delaunay=True,radius=(2.5,50))
    sq.gr.nhood_enrichment(adata_s, cluster_key=cluster_key)

    NN_score = pd.DataFrame(
        data=adata_s.uns[f'{cluster_key}_nhood_enrichment']['zscore'], 
        index=adata_s.obs[cluster_key].cat.categories, 
        columns=adata_s.obs[cluster_key].cat.categories,
        )
    if remove_negative_values:
        NN_score[NN_score < 0] = 0

    if direction == -1:
        NN_score = NN_score.T
         
    xy = adata_s.obsm['spatial']
    x = xy[:,0]
    y = xy[:,1]
    c = adata_s.obs[cluster_key]

    df , coords = _hexbin_make(x, y, c,spacing = spacing, min_count=min_count, n_jobs=-1)
    tree = KDTree(coords)
    #get the nearest neighbors
    vector_field = []
    for i, coord in enumerate(coords):
        dist, ind = tree.query(coord, k=25)
        ind = ind[dist < spacing*2]
        if len(ind) < 10:
            vector_field.append(np.array([0,0]))
        else:
            dist, ind = tree.query(coord, k=k*7)
            ind = ind[dist < spacing*2]
            vectors = normalize(np.array([coords[ind[0]] - coords[i] for i in ind]))[1:]
            df_nn = df.iloc[:,ind]
            
            weights = []
            for i in range(1,df_nn.shape[1]):
                a = NN_score.values@df_nn.iloc[:,0].values
                b = NN_score.values@df_nn.iloc[:,i].values
                v = np.dot(a,b)
                weights.append(v)
            weights = np.array(weights)
            vectors *= np.array(weights)[:,None]
            vector = vectors.sum(axis=0)
            vector_field.append(vector)

    vector_field = np.array(vector_field)
    #vector_field = normalize(vector_field) * spacing
    adata_s.uns['vector_field_delta'] = vector_field
    adata_s.uns['vector_field_origin'] = coords

    if copy:
        return adata_s


def _hexbin_make(
    x,
    y,
    c,
    spacing: float=100, 
    min_count: int = 2, 
    feature_selection: np.ndarray=None,              
    n_jobs: int=-1) -> Tuple[Any, np.ndarray]:
        """
        Bin 2D point data with hexagonal bins.
        
        Stores the centroids of the hexagons under hexagon_coordinates,
        and the hexagon shape under hexbin_hexagon_shape.
        Args:
            spacing (float): distance between tile centers, in same units as 
                the data. The function makes hexagons with the point up: ⬡
            min_count (int): Minimal number of molecules in a tile to keep the 
                tile in the dataset.
            n_jobs (int, optional): Number of workers. If -1 it takes the max
                cpu count. Defaults to -1.
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: 
            Pandas Dataframe with counts for each valid tile.
            Numpy Array with centroid coordinates for the tiles.
            
        """        
        import multiprocessing
        #workers
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
            
        df = pd.DataFrame({'x':x, 'y':y, 'c':c})
        #Get canvas size
        max_x = x.max()
        min_x = x.min()
        max_y = y.max()
        min_y = y.min()
        
        x_extent = max_x - min_x
        y_extent = max_y - min_y 

        
        #Find X range 
        n_points_x = math.ceil((max_x - min_x) / spacing)
        #Correct x range to match whole number of tiles
        full_x_extent = n_points_x * spacing
        difference_x = full_x_extent - x_extent
        min_x = min_x - (0.5 * difference_x)
        max_x = max_x + (0.5 * difference_x)
        
        #Find Y range 
        y_spacing = (spacing * np.sqrt(3)) / 2
        n_points_y = math.ceil((max_y - min_y) / y_spacing)
        #Correct y range to match whole number of tiles
        full_y_extent = n_points_y * y_spacing
        difference_y = full_y_extent - y_extent
        min_y = min_y - (0.5 * difference_y)
        max_y = max_y + (0.5 * difference_y)
            
        #make hexagonal grid
        x = np.arange(min_x, max_x, spacing, dtype=float)
        y = np.arange(min_y, max_y, y_spacing, dtype=float)
        xx, yy = np.meshgrid(x, y)
        #Offset every second row 
        xx[::2, :] += 0.5*spacing
        coordinates = np.array([xx.ravel(), yy.ravel()]).T
        
        #make KDTree
        tree = KDTree(coordinates)
        
        #classes

        classes = np.unique(c)
        #Make Results dataframe
        n_classes = len(classes)
        n_tiles = coordinates.shape[0]
        df_hex = pd.DataFrame(data=np.zeros((n_classes, n_tiles)),
                            index=classes, 
                            columns=[j for j in range(n_tiles)])
        
        #Hexagonal binning of data
        for i, c in enumerate(classes):
            data = df[df['c'] == c]
            data = data.loc[:,['x', 'y']].values
            #Query nearest neighbour ()
            dist, idx = tree.query(data, distance_upper_bound=spacing, workers=n_jobs)
            #Count the number of hits
            count = np.zeros(n_tiles)
            counter = Counter(idx)
            count[list(counter.keys())] = list(counter.values())
            #Add to dataframe
            df_hex.loc[c] = count
        
        #make hexagon coordinates
        hexbin_hexagon_shape = _hexagon_shape(spacing, closed=True)
            
        #Filter on number of counts
        filt = df_hex.sum() >= min_count
        df_hex = df_hex.loc[:, filt]
        coordinates = coordinates[filt]
        hexbin_coordinates = coordinates
        
        #store settings for plotting
        _hexbin_params = f'spacing_{spacing}_min_count_{min_count}_nclasses{n_classes}'

        return df_hex, coordinates
    
def _hexagon_shape(spacing: float, closed=True) -> np.ndarray:
        """Make coordinates of hexagon with the point up.      
        Args:
            spacing (float): Distance between centroid and centroid of 
                horizontal neighbour. 
            closed (bool, optional): If True first and last point will be
                identical. Needed for some polygon algorithms. 
                Defaults to True.
        Returns:
            [np.ndarray]: Array of coordinates.
        """
        
        # X coodrinate 4 corners
        x = 0.5 * spacing
        # Y coordinate 4 corners, tan(30)=1/sqrt(3)
        y = 0.5 * spacing *  (1 / np.sqrt(3))
        # Y coordinate top/bottom
        h = (0.5 * spacing) / (np.sqrt(3)/2)
        
        coordinates = np.array([[0, h],
                                [x, y,],
                                [x, -y],
                                [0, -h],
                                [-x, -y],
                                [-x, y]])
        
        if closed:
            coordinates = np.vstack((coordinates, coordinates[0]))
            
        return coordinates