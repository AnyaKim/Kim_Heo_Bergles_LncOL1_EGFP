# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 14:19:35 2026

@author: jacob
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:20:03 2026

@author: jacob
"""
#%%

import os

os.chdir("C:/Users/jacob/Desktop/pwd/Allen_AnyaDotplot")

import pandas as pd
from pathlib import Path
import numpy as np
import anndata
import time
import matplotlib.pyplot as plt
import pickle as pkl
import os



import z5py

import glob, os
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


import tifffile as tiff

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

from skimage.measure import label, regionprops, regionprops_table

import json
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'    

import anndata as ad

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc
import plotnine as pn

#%%

adata = sc.read_h5ad('Allen_AnyaDotplot.h5ad')
adata = adata[adata.obs['age'] == '4wk'].copy()
#adata.X = adata.raw.X
#adata.layers['raw_counts'] = adata.X.copy()  # store raw counts
#sc.pp.normalize_total(adata, target_sum=1e4)
#sc.pp.log1p(adata)
#adata.layers['log1p'] = adata.X.copy()  # store log-normalized data
#adata.raw = adata
#sc.pp.scale(adata, max_value=10)


obs = adata.obs

adata.obs['broad_cell_type'] = adata.obs['clust_annot'].str.split('-', n=1).str[0]
obs = adata.obs


mask = adata.obs['broad_cell_type'] == 'MSN'
adata.obs.loc[mask, 'broad_cell_type'] = adata.obs.loc[mask, 'clust_annot']

mask = adata.obs['broad_cell_type'].isin(['MSN-D1-1', 'MSN-D1-2'])
adata.obs.loc[mask, 'broad_cell_type'] = 'MSN-D1'

mask = adata.obs['broad_cell_type'].isin(['OPC', 'Olig'])
adata.obs.loc[mask, 'broad_cell_type'] = 'OL Lineage'


clusters = pd.DataFrame(pd.unique(adata.obs['broad_cell_type']))


desired_types = ['OL Lineage', 'MSN-D1', 'MSN-D2', 'Astro', 'Micro', 'Endo', 'ExN', 'InN']
adata_subset = adata[adata.obs['broad_cell_type'].isin(desired_types)]


markers = 'ENSMUSG00000115529'
sc.set_figure_params(dpi=300, format='svg', dpi_save=300, figsize=(12, 4))

fraction_expressing_order = ["Endo", "Astro", "Micro", "InN", "ExN", "MSN-D1", "MSN-D2", "OL Lineage"]



sc.pl.dotplot(adata_subset, markers, groupby = 'broad_cell_type', categories_order = fraction_expressing_order, use_raw=False, save='20261003_Allen96RikDotplot_4wOnly_Renormalize.svg')






#%%


adata_subset.X = adata_subset.raw.X
pseudobulked = sc.get.aggregate(adata_subset, by=['broad_cell_type', 'donor_id'], func="sum")    
sums = pd.DataFrame(pseudobulked.layers["sum"])
gene_names = adata.var.index
sums.columns = gene_names


# Run differential expression
deseq_object = DeseqDataSet(counts = sums, metadata=pd.DataFrame(pseudobulked.obs), design_factors = "broad_cell_type")

deseq_object.fit_size_factors()
deseq_object.fit_genewise_dispersions()
deseq_object.fit_dispersion_trend()
deseq_object.fit_dispersion_prior()
deseq_object.fit_MAP_dispersions()
deseq_object.fit_LFC()

deseq_object.calculate_cooks()
if deseq_object.refit_cooks:
    # Replace outlier counts
    deseq_object.refit()
    
    



from itertools import combinations

cell_types = adata_subset.obs['broad_cell_type'].unique()
results = pd.DataFrame()
    
for ct1, ct2 in combinations(cell_types, 2):

    stat_res = DeseqStats(deseq_object, contrast=['broad_cell_type', ct1, ct2], alpha=0.05, cooks_filter=True, independent_filter=True)
    stat_res.run_wald_test()
    stat_res.p_values
    stat_res._p_value_adjustment()
    stat_res.summary()
    
    results_df = stat_res.results_df
    results_df['cell_type_1'] = ct1
    results_df['cell_type_comparison'] = ct2
    lncOL1_result = pd.DataFrame(results_df.loc['ENSMUSG00000115529']).transpose()
    results = pd.concat([results, lncOL1_result])




stat_res = DeseqStats(deseq_object, contrast=['broad_cell_type', 'MSN-D1', 'MSN-D2'], alpha=0.05, cooks_filter=True, independent_filter=True)
stat_res.run_wald_test()
stat_res.p_values
stat_res._p_value_adjustment()
stat_res.summary()

results_df = stat_res.results_df
lncOL1 = results_df.loc['ENSMUSG00000115529']

results.to_csv('20261003_Pairwise_96Rik_AllenDataset_DESeq2.csv')

#%%


def get_expression_proportions(adata, genes, groupby='cell_type'):

    
    results = []
    
    for cell_type in adata.obs[groupby].unique():
        # Subset to cell type
        mask = adata.obs[groupby] == cell_type
        subset = adata[mask, :]
        
        for gene in genes:
            if gene in adata.var_names:
                # Get expression values
                expr = subset[:, gene].X
                
                # Handle sparse matrices
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                else:
                    expr = np.array(expr).flatten()
                
                # Calculate metrics
                n_cells = len(expr)
                n_expressing = np.sum(expr > 0)
                pct_expressed = (n_expressing / n_cells) * 100
                mean_expr = np.mean(expr[expr > 0]) if n_expressing > 0 else 0
                mean_expr_all = np.mean(expr)
                
                results.append({
                    'cell_type': cell_type,
                    'gene': gene,
                    'pct_expressed': pct_expressed,
                    'mean_expr_in_expressing': mean_expr,
                    'mean_expr_all_cells': mean_expr_all,
                    'n_cells': n_cells,
                    'n_expressing': n_expressing
                })
    
    return pd.DataFrame(results)


desired_types = ['OL Lineage', 'MSN-D1', 'MSN-D2', 'Astro', 'Micro', 'Endo', 'ExN', 'InN']
adata_subset = adata[adata.obs['broad_cell_type'].isin(desired_types)]

# Usage
genes_of_interest = ['ENSMUSG00000115529']
df = get_expression_proportions(adata_subset, genes_of_interest, groupby='broad_cell_type')
print(df)

sc.pl.dotplot(adata_subset, markers, groupby = 'broad_cell_type')


df.to_csv('20260406_96Rik_ExpressionByCellType.csv')
