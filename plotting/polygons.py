import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import logging
import os
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib import colors as mcolors
import random

logging.basicConfig(level=logging.INFO)

#rcParams['mathtext.rm'] = 'Arial'
#rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 7


def random_color():
	return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])


def _inside(p, xlim, ylim):
    if p.centroid.coords[0][0] >= xlim[0] and p.centroid.coords[0][0] < xlim[1] and p.centroid.coords[0][1] >= ylim[0] and p.centroid.coords[0][1] < ylim[1]:
        return True
    else:
        return False
    

def plot_polygons(
		adata:sc.AnnData,
		sample:str,
		clusters:list,
		palette:dict,
		geometry_key:str = 'Polygons',
		xlim:tuple = None, #= (11000, 13000)
		ylim:tuple = None, #= (5000, 7000)
		cluster_key:str = 'CombinedNameMergeImmune',
		grey_clusters:list = [],
		gray_color:str = '#ececec',
		annotate:bool = False, # Will write arrow and legend on top of one of each polygon clusters
		annotation_loc:dict = {}, # Which iloc to use for annotation {cluster:iloc}
		area_min_size:int = 25, # Minimum area size of polygons to plot
		facecolor:tuple = (1,1,1), #background, defaults white
		figsize:tuple = (10,10),
		alpha:float = 0.75,
		alpha_gray:float = 0.25,
		linewidth_gray:float = 0.05,
		fontsize:int = 8,
		show_axis:bool=False,
		save:bool=False,
		savepath:str = None,
		image_downscale:int=5, # defaults to 5 because our HEs are 5x downsampled
		show_scalebar:bool=True,
		image:np.array=None,
		flipy:bool=False,
		flipx:bool=False,
		markersize=1,
		linewidth=0.05,
		annotation_rotation:int=0,
		annotation_text_offset:tuple=(-50,0),
		dpi=300,
		ax=None,
		):

	scale_factor = 1
	adata = adata[adata.obs['Sample'] == sample]
	adata = adata[(adata.obs['Area'] > area_min_size), :]
	adata = adata[adata.obs[cluster_key].isin(clusters + grey_clusters), :]
	logging.info('First filter, {} cells left'.format(adata.shape[0]))

	polygons = adata.obs[geometry_key]
	ispoint = polygons.values[0].count('POINT')
	geometry = gpd.GeoSeries.from_wkt(polygons)

	if image is not None:
		scale_factor = 0.27*image_downscale
		geometry = geometry.affine_transform([1/scale_factor, 0, 0, 1/scale_factor, 0, 0])

	if type(alpha) == str:
		alpha = adata.obsm[alpha]
	else:
		alpha = np.ones(adata.shape[0])*alpha

	gdf = gpd.GeoDataFrame(geometry=geometry,
		data=
			{
				cluster_key:adata.obs[cluster_key],
				'Area':adata.obs['Area'],
				'color':[palette[p] for p in adata.obs[cluster_key]],
				'alpha':alpha,
		}
	)
	if xlim is not None and ylim is not None:
		logging.info('Selecting cells in zoom area')
		gdf = gdf[gdf.loc[:,'geometry'].apply(lambda p: _inside(p, xlim=xlim, ylim=ylim))]
		translated_geom = gdf.loc[:,'geometry'].translate(xoff=-xlim[0], yoff=-ylim[0])
		gdf.loc[:,'geometry'] = translated_geom
	logging.info('Zoom filter, {} cells left'.format(gdf.shape[0]))

	gdf_col = gdf[gdf[cluster_key].isin(clusters)]

	if ax is None:
		fig, ax1 = plt.subplots(figsize=figsize)
	else:
		ax1 = ax
	ax1.set_facecolor(facecolor)
	if image is not None:
		if flipy:
			image = np.flipud(image)
		if flipx:
			image = np.fliplr(image)
		if xlim is None:
			xlim = (0, image.shape[1])
		if ylim is None:
			ylim = (0, image.shape[0])
		ax1.imshow(image[ylim[0]:ylim[1], xlim[0]:xlim[1]])
	

	if ispoint:
		_ = gdf[gdf[cluster_key].isin(grey_clusters)].plot(
			color=gray_color,
			markersize=markersize, 
			edgecolor='black',
			linewidth=linewidth_gray,
			ax=ax1,
			rasterized=True,
			facecolor=facecolor, 
			alpha=alpha_gray,
			)
	else:
		_ = gdf[gdf[cluster_key].isin(grey_clusters)].plot(
			color=gray_color,
			edgecolor='black',
			linewidth=linewidth_gray,
			ax=ax1,
			rasterized=True,
			facecolor=facecolor, 
			alpha=alpha_gray,
			)
		
	for c in clusters:
		gdf_ = gdf_col[gdf_col[cluster_key] == c]
		if type(alpha) == str:
			alpha = gdf_.alpha.values
		if ispoint:
			im = gdf_.plot(
				color= palette[c], 
				markersize=markersize, 
				#edgecolor='black',
				facecolor=palette[c],
				linewidth=linewidth,
				ax=ax1,
				rasterized=True,
				alpha=alpha)
		else:
			im = gdf_.plot(color= palette[c], edgecolor='black',linewidth=linewidth, ax=ax1,rasterized=True,facecolor=facecolor,alpha=alpha)
	ax1.set_rasterization_zorder(0)
	if annotate:
		ann = gdf_col[(gdf_col.Area > 200) & (gdf_col.Area <= 1000)]
		for m in ann[cluster_key].unique():
			i = annotation_loc[m] if m in annotation_loc.keys() else 0
			print(i)
			x = ann[ann[cluster_key] == m].iloc[i,:]
			ax1.annotate(
				text='{}'.format(x[cluster_key]), 
				fontsize=fontsize,
				xy=x.geometry.centroid.coords[0], 
				ha='center', 
				alpha=0.9, 
				rotation=annotation_rotation,
				color='black' if facecolor == (1,1,1) else 'white',
				xytext=x.geometry.centroid.coords[0]+np.array(annotation_text_offset), 
			)
			ax1.arrow(
				x = x.geometry.centroid.coords[0][0]-100,
				y = x.geometry.centroid.coords[0][1]-100,
				dx = 10,
				dy = 10,
				overhang=0,
				head_starts_at_zero=True,
				head_width=50,
				color='black' if facecolor == (1,1,1) else 'white',
			)

	if show_scalebar:
		scalebar = ScaleBar(
			scale_factor,
			units='um',
			length_fraction=.1,
			location='lower right'
		) # 1 pixel = 0.2 meter
		plt.gca().add_artist(scalebar)

	plt.tight_layout()

	if show_axis == False:
		ax1.spines[['left','right', 'top','bottom']].set_visible(False)
		ax1.set_xticks([])
		ax1.set_yticks([])
		#ax1.axis('off')
	
	if ax is None:
		if save:
			if savepath is None:
				savepath = os.path.join('figures','{}_zoom{}.svg'.format(sample, clusters))
			transparent = True if facecolor == (1,1,1) else False
			plt.savefig(savepath, dpi=dpi, transparent=transparent,bbox_inches='tight')
		
		plt.show()


def plot_polygons_expression(
		adata:sc.AnnData,
		sample:str,
		genes:list,
		cmap = 'magma',
		bgval = 0, # background value for plotting expression
		mquant = 0.99, # max quantile for plotting expression
		plot_grays:bool = True,
		palette:dict=None,
		geometry_key:str = 'Polygons',
		xlim:tuple = None, #= (11000, 13000)
		ylim:tuple = None, #= (5000, 7000)
		cluster_key:str = 'CombinedNameMergeImmune',
		annotate:bool = False, # Will write arrow and legend on top of one of each polygon clusters
		annotation_loc:dict = {}, # Which iloc to use for annotation {cluster:iloc}
		area_min_size:int = 25, # Minimum area size of polygons to plot
		facecolor:tuple = (1,1,1), #background, defaults white
		figsize:tuple = (10,10),
		alpha:float = 0.75,
		alpha_gray:float = 0.75,
		fontsize:int = 8,
		show_axis:bool=False,
		save:bool=False,
		savepath:str = None,
		image_downscale:int=5, # defaults to 5 because our HEs are 5x downsampled
		show_scalebar:bool=True,
		image:np.array=None,
		flipy:bool=False,
		flipx:bool=False,
		annotation_rotation:int=0,
		annotation_text_offset:tuple=(-50,0),
		ax=None,
		dpi=300,
		):

	scale_factor = 1
	adata = adata[adata.obs['Sample'] == sample]
	adata = adata[(adata.obs['Area'] > area_min_size), :]
	expression = adata[:,genes].X.mean(axis=1).toarray()
	expression = np.clip(expression, bgval, np.quantile(expression, mquant))
	logging.info('First filter, {} cells left'.format(adata.shape[0]))

	polygons = adata.obs[geometry_key]
	gray_color = '#ececec'
	geometry = gpd.GeoSeries.from_wkt(polygons)

	if image is not None:
		scale_factor = 0.27*image_downscale
		geometry = geometry.affine_transform([1/scale_factor, 0, 0, 1/scale_factor, 0, 0])

	gdf = gpd.GeoDataFrame(geometry=geometry,
		data=
			{
				cluster_key:adata.obs[cluster_key],
				'Expression': expression,
				'Area':adata.obs['Area'],
		}
	)
	if xlim is not None and ylim is not None:
		logging.info('Selecting cells in zoom area')
		gdf = gdf[gdf.loc[:,'geometry'].apply(lambda p: _inside(p, xlim=xlim, ylim=ylim))]
		translated_geom = gdf.loc[:,'geometry'].translate(xoff=-xlim[0], yoff=-ylim[0])
		gdf.loc[:,'geometry'] = translated_geom
	logging.info('Zoom filter, {} cells left'.format(gdf.shape[0]))

	#gdf_col = gdf[gdf[cluster_key].isin(clusters)]

	if ax is None:
		fig, ax1 = plt.subplots(figsize=figsize)
	else:
		ax1 = ax
	ax1.set_facecolor(facecolor)
	if image is not None:
		if flipy:
			image = np.flipud(image)
		if flipx:
			image = np.fliplr(image)
		ax1.imshow(image[ylim[0]:ylim[1], xlim[0]:xlim[1]])

	if plot_grays:
		im_gray = gdf[gdf['Expression'] <= bgval ].plot(color=gray_color,edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=facecolor, alpha=alpha_gray)
	gdf_col = gdf[gdf['Expression'] > bgval ]

	order = np.argsort(gdf_col['Expression'])
	gdf_col = gdf_col.iloc[order]
	last = bgval
	im = gdf_col.plot(column='Expression', cmap=cmap, edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=facecolor,alpha=alpha)

	if show_scalebar:
		scalebar = ScaleBar(
			scale_factor,
			units='um',
			length_fraction=.1,
			location='lower right'
		) # 1 pixel = 0.2 meter
		plt.gca().add_artist(scalebar)

	plt.tight_layout()

	if show_axis == False:
		ax1.spines[['left','right', 'top','bottom']].set_visible(False)
		ax1.set_xticks([])
		ax1.set_yticks([])
		#ax1.axis('off')
	
	if ax is None:
		if save:
			if savepath is None:
				savepath = os.path.join('figures','{}_zoom{}.svg'.format(sample, clusters))
			transparent = True if facecolor == (1,1,1) else False
			plt.savefig(savepath,dpi=dpi,format='svg', transparent=transparent,bbox_inches='tight')
		
		plt.show()


'''def polygons_gene_ratio(
		adata:sc.AnnData,
		sample:str,
		geneA:list,
		geneB:list,
		clusters:list,
		cluster_key:str = 'CombinedNameMerge',
		cmap = 'coolwarm',
		bgval = 0, # background value for plotting expression
		mquant = 0.99, # max quantile for plotting expression
		palette:dict=None,
		geometry_key:str = 'Polygons',
		xlim:tuple = None, #= (11000, 13000)
		ylim:tuple = None, #= (5000, 7000)

		area_min_size:int = 25, # Minimum area size of polygons to plot
		facecolor:tuple = (1,1,1), #background, defaults white
		figsize:tuple = (10,10),
		alpha:float = 0.75,
		alpha_gray:float = 0.75,
		fontsize:int = 8,
		show_axis:bool=False,
		save:bool=False,
		savepath:str = None,
		image_downscale:int=5, # defaults to 5 because our HEs are 5x downsampled
		show_scalebar:bool=True,
		image:np.array=None,
		flipy:bool=False,
		flipx:bool=False,
		annotation_rotation:int=0,
		annotation_text_offset:tuple=(-50,0),
		ax=None,
		dpi=300,
		):

	scale_factor = 1
	adata = adata[adata.obs['Sample'] == sample]
	adata = adata[(adata.obs['Area'] > area_min_size), :]
	adata = adata[adata.obs[cluster_key].isin(clusters)]

	expressionA = adata[:,geneA].X.toarray()
	expressionB = adata[:,geneB].X.toarray()

	ratio = np.divide(expressionA, expressionB)

	#expression = np.clip(expression, bgval, np.quantile(expression, mquant))
	logging.info('First filter, {} cells left'.format(adata.shape[0]))

	polygons = adata.obs[geometry_key]
	gray_color = '#ececec'
	geometry = gpd.GeoSeries.from_wkt(polygons)

	if image is not None:
		scale_factor = 0.27*image_downscale
		geometry = geometry.affine_transform([1/scale_factor, 0, 0, 1/scale_factor, 0, 0])

	gdf = gpd.GeoDataFrame(geometry=geometry,
		data=
			{
				cluster_key:adata.obs[cluster_key],
				'Expression': expression,
				'Area':adata.obs['Area'],
		}
	)
	if xlim is not None and ylim is not None:
		logging.info('Selecting cells in zoom area')
		gdf = gdf[gdf.loc[:,'geometry'].apply(lambda p: _inside(p, xlim=xlim, ylim=ylim))]
		translated_geom = gdf.loc[:,'geometry'].translate(xoff=-xlim[0], yoff=-ylim[0])
		gdf.loc[:,'geometry'] = translated_geom
	logging.info('Zoom filter, {} cells left'.format(gdf.shape[0]))

	#gdf_col = gdf[gdf[cluster_key].isin(clusters)]

	if ax is None:
		fig, ax1 = plt.subplots(figsize=figsize)
	else:
		ax1 = ax
	ax1.set_facecolor(facecolor)
	if image is not None:
		if flipy:
			image = np.flipud(image)
		if flipx:
			image = np.fliplr(image)
		ax1.imshow(image[ylim[0]:ylim[1], xlim[0]:xlim[1]])

	im_gray = gdf[gdf['Expression'] <= bgval ].plot(color=gray_color,edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=facecolor, alpha=alpha_gray)
	gdf_col = gdf[gdf['Expression'] > bgval ]

	order = np.argsort(gdf_col['Expression'])
	gdf_col = gdf_col.iloc[order]
	last = bgval
	im = gdf_col.plot(column='Expression', cmap=cmap, edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=facecolor,alpha=alpha)

	if show_scalebar:
		scalebar = ScaleBar(
			scale_factor,
			units='um',
			length_fraction=.1,
			location='lower right'
		) # 1 pixel = 0.2 meter
		plt.gca().add_artist(scalebar)

	plt.tight_layout()

	if show_axis == False:
		ax1.spines[['left','right', 'top','bottom']].set_visible(False)
		ax1.set_xticks([])
		ax1.set_yticks([])
		#ax1.axis('off')
	
	if ax is None:
		if save:
			if savepath is None:
				savepath = os.path.join('figures','{}_zoom{}.svg'.format(sample, clusters))
			transparent = True if facecolor == (1,1,1) else False
			plt.savefig(savepath,dpi=dpi,format='svg', transparent=transparent,bbox_inches='tight')
		
		plt.show()'''