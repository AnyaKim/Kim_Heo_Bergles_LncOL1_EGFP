# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:34:42 2025

@author: jacob
"""

import z5py

import glob, os
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import pandas as pd

import tifffile as tiff

import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean

from skimage.measure import label, regionprops, regionprops_table

import json
import matplotlib.pyplot as plt

### Makes it so that svg exports text as editable text!
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'    

os.chdir("C:/Users/jacob/Desktop/pwd/Anya_NFO_ISH")
from postprocess_func import *
import seaborn as sns
from scipy import stats   
from adjustText import adjust_text

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import sys
sys.path.append("..")

from get_brain_metadata import *

from PLOT_FUNCTIONS_postprocess_compare import *

import pickle as pkl
os.chdir("C:/Users/jacob/Desktop/pwd/Anya_NFO_ISH")


NFO_isocx = open('./20260102_correlation_ISH_energy_NFO_Density_Isocx', 'rb')
NFO_isocx = pkl.load(NFO_isocx)

    
#%%
def volcano_plot_from_corr(df, xcol='correlation', pcol='p_adj',
                           thresh_pval=0.05, thresh_x=0.25,
                           xlim=1, ylim=20, figsize=(3,3),
                           fontsize=14, genes_to_plot=None, num_genes=3, 
                           label_fontsize=10, file_name="volcano_plot.svg"):

    df = df.copy()
    df['-log10pval'] = -np.log10(df[pcol])

    def get_change_type(row):
        if row[pcol] < thresh_pval and row[xcol] > 0:
            return 'Up'
        elif  row[pcol] < thresh_pval and row[xcol] < 0:
            return 'Down'
        else:
            return 'None'
    df['change_type'] = df.apply(get_change_type, axis=1)

    # Color target genes
    def get_gene_color(gene):
        gene_str = str(gene)
        gene_lower = gene_str.lower()
        if gene not in genes_to_plot:
            return 'gray'
        if gene_lower.startswith('bmp'):
            return '#51FD00'
        elif gene_lower.startswith('wnt'):
            return '#FFFF00'
        elif gene_lower in ['gas6', 'pvalb', 'syt2']:
            return '#FF00E7'
        elif gene_lower in ['mobp', 'mog']:
            return 'tab:red'
        elif gene_lower in ['syt17', 'lingo2']:
            return 'tab:blue'

    df['gene_color'] = df['gene_name'].apply(get_gene_color)

    # Split into gray (background) and colored (highlighted) points
    gray_df = df[df['gene_color'] == 'gray']
    colored_df = df[df['gene_color'] != 'gray']

    plt.figure(figsize=figsize)

    # Plot gray points first (background, lower zorder)
    ax = sns.scatterplot(data=gray_df, x=xcol, y='-log10pval', color='gray',
                         s=30, linewidth=0, rasterized=True, clip_on=False,
                         zorder=1, alpha=0.5)

    # Plot colored points on top (higher zorder)
    sns.scatterplot(data=colored_df, x=xcol, y='-log10pval', color=colored_df['gene_color'],
                    s=30, linewidth=0, rasterized=True, clip_on=False,
                    zorder=2, ax=ax, alpha=0.75)

    # Add threshold lines
    ax.axhline(-np.log10(thresh_pval), linestyle='--', color='black', linewidth=1, zorder=3)
    ax.axvline(thresh_x, linestyle='--', color='black', linewidth=1, zorder=3)
    ax.axvline(-thresh_x, linestyle='--', color='black', linewidth=1, zorder=3)

    # Determine which genes to label
    if genes_to_plot == 'significant_cell_types':
        top_genes = df[df[pcol] < 0.05]
    elif genes_to_plot is not None:
        top_genes = df[df['gene_name'].isin(genes_to_plot)]
    else:
        up_genes = df[df['change_type'] == 'Up'].nlargest(num_genes, xcol)
        down_genes = df[df['change_type'] == 'Down'].nsmallest(num_genes, xcol)
        top_genes = pd.concat([up_genes, down_genes])

    # Sort genes by y-value within each group to prevent overlaps
    up_labeled = top_genes[top_genes['change_type'] == 'Up'].sort_values('-log10pval')
    down_labeled = top_genes[top_genes['change_type'] == 'Down'].sort_values('-log10pval')
    other_labeled = top_genes[top_genes['change_type'] == 'None'].sort_values('-log10pval')

    # Function to adjust y positions to prevent overlap
    def adjust_positions(genes_subset, min_spacing=3.5):
        """Adjust y positions to ensure minimum spacing between labels"""
        if len(genes_subset) == 0:
            return []
        
        positions = []
        sorted_genes = genes_subset.sort_values('-log10pval').copy()
        
        for idx, (_, row) in enumerate(sorted_genes.iterrows()):
            y = row['-log10pval']
            
            # Check for overlaps with previous positions
            if positions:
                last_y = positions[-1]
                if y - last_y < min_spacing:
                    y = last_y + min_spacing
            
            positions.append(y)
        
        return list(zip(sorted_genes.index, positions))

    # Annotate genes with adjusted positions and longer lines
    for gene_group in [up_labeled, down_labeled, other_labeled]:
        if len(gene_group) == 0:
            continue
            
        adjusted = adjust_positions(gene_group, min_spacing=3.5)
        
        for orig_idx, adjusted_y in adjusted:
            row = gene_group.loc[orig_idx]
            x, orig_y = row[xcol], row['-log10pval']
            gene = row.get('gene_name', '')

            if row['change_type'] == 'Up':
                offset_x = -0.55  # Increased from 0.15
                ha = 'left'
            elif row['change_type'] == 'Down':
                offset_x = 0.55  # Increased from -0.15
                ha = 'right'
            else:
                offset_x = 0
                ha = 'center'

            # Calculate offset_y toensure that points do not overlap
            offset_y = max(1.0, abs(adjusted_y - orig_y))
            
            ax.annotate(gene,
                        xy=(x, orig_y),
                        xytext=(x + offset_x, adjusted_y),
                        textcoords='data',
                        ha=ha, va='center',
                        fontsize=label_fontsize,
                        color='black',
                        arrowprops=dict(arrowstyle='-', color='black', lw=0.5, 
                                      shrinkA=0, shrinkB=0),
                        zorder=4)

    plt.xlim([-xlim, xlim])
    plt.ylim([0, ylim])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Correlation', fontsize=fontsize)
    plt.ylabel('-Log10 (Adjusted P-Value)', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    plt.savefig(file_name, format='svg', dpi=300)
    plt.show()    


volcano_plot_from_corr(NFO_isocx, xcol='correlation', pcol='p_adj',
                           thresh_pval=0.05, thresh_x=0.156142,
                           xlim=1, ylim=40, figsize=(4,4),
                           fontsize=14, genes_to_plot=['Bmp8', 'Bmp6', 'Bmp10', 'Bmp2k', 'Bmp2', 'Bmp1', 'Wnt2', 'Wnt5a', 'Wnt5b', 'Gas6', 'Pvalb', 'Syt2', 'Syt17', 'Lingo2', 'Mobp', 'Mog'], 
                           label_fontsize=10, file_name="20261005_Anya_BMP_Wnt_PV_Volcano.svg")

