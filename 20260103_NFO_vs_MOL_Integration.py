# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 15:51:05 2026

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

import pickle as pkl

import sys
sys.path.append("..")

from get_brain_metadata import *

from PLOT_FUNCTIONS_postprocess_compare import *

#%%

plot_all = False
import pickle as pkl


fontsize = 14

fig_stand = np.asarray([4.5, 5])
fig_long = np.asarray([4.8,4])

P60_x_lim = 18000

df_concat_NFO = open('./NFO_Dataframe.pkl', 'rb')
df_concat_NFO = pkl.load(df_concat_NFO)
unique_experiments_NFO = list(df_concat_NFO['exp'].unique())


pooled_df_NFO = df_concat_NFO[df_concat_NFO['exp'].isin(['LncOL1_P30'])]
df_means_NFO = pooled_df_NFO.groupby(pooled_df_NFO['acronym']).mean(numeric_only=True)

df_non_numeric_NFO = pooled_df_NFO.groupby(pooled_df_NFO['acronym']).first()
df_numeric_means_NFO = pooled_df_NFO.groupby(pooled_df_NFO['acronym']).mean(numeric_only=True)
df_means_NFO = df_numeric_means_NFO.join(df_non_numeric_NFO.select_dtypes(exclude='number'))


#%%

os.chdir("C:/Users/jacob/Desktop/pwd/TigerISH_OL_Atlas")


df_concat_MOL = open('./df_concat_ish', 'rb')
df_concat_MOL = pkl.load(df_concat_MOL)
unique_experiments_MOL = list(df_concat_MOL['exp'].unique())


#pooled_df_MOL = df_concat_MOL[df_concat_MOL['exp'].isin(['P60', 'P60_F'])]
pooled_df_MOL = df_concat_MOL[df_concat_MOL['exp'].isin(['P240'])]
df_means_MOL = pooled_df_MOL.groupby(pooled_df_MOL.index).mean(numeric_only=True)

df_non_numeric_MOL = pooled_df_MOL.groupby(pooled_df_MOL.index).first()
df_numeric_means_MOL = pooled_df_MOL.groupby(pooled_df_MOL.index).mean(numeric_only=True)
df_means_MOL = df_numeric_means_MOL.join(df_non_numeric_MOL.select_dtypes(exclude='number'))





#%%
gray_idx = get_sub_regions_atlas(df_means_NFO, child_id=[], sub_keys=[], reg_name = 'Basic cell groups and regions')
df_means_NFO = df_means_NFO.iloc[gray_idx]
striatum_idx = get_sub_regions_atlas(df_means_NFO, child_id=[], sub_keys=[], reg_name = 'Striatum')
df_means_NFO = df_means_NFO.drop(df_means_NFO.index[striatum_idx])


NFO_indexed = df_means_NFO.set_index('ids')
MOL_indexed = df_means_MOL.set_index('ids')
#df1_indexed = df1_indexed.loc[df2_indexed.index]



common_idx = NFO_indexed.index.intersection(MOL_indexed.index)
NFO_indexed = NFO_indexed.loc[common_idx]

common_idx = MOL_indexed.index.intersection(NFO_indexed.index)
MOL_indexed = MOL_indexed.loc[common_idx]


MOL_indexed.rename(columns={'density_W': 'Density_MOL'}, inplace=True)
NFO_indexed.rename(columns={'density_W': 'Density_NFO'}, inplace=True)

result = pd.merge(NFO_indexed, MOL_indexed, left_index=True, right_index=True, how='outer')
result = result[result['children_y'].str.len() == 0]


#%%

os.chdir("C:/Users/jacob/Desktop/pwd/Anya_NFO_ISH")

def correlation_plot(df, col1, col2, output_png, color='blue', fontsize=12, 
                    label_col='acronym', labels_to_show=None):
    
    # Create the plot
    plt.figure(figsize=(7.5, 4.5))
    ax = sns.regplot(x=col1, y=col2, data=df, scatter_kws={'color': color}, line_kws={'color': color})
    
    # Add labels only for specified regions
    if labels_to_show is not None:
        for idx, row in df.iterrows():
            if row[label_col] in labels_to_show:
                ax.annotate(row[label_col], 
                           xy=(row[col1], row[col2]),  # point location
                           xytext=(20, 20),  # larger offset in points
                           textcoords='offset points',
                           fontsize=fontsize - 4,
                           alpha=0.8,
                           arrowprops=dict(arrowstyle='-', 
                                         color='black', 
                                         lw=1,
                                         alpha=0.6))
    
    # Set the labels and customize the appearance
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.xlabel(col1, fontsize=fontsize)
    plt.ylabel(col2, fontsize=fontsize)
    
    # Customize the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_png, format='svg', dpi=300)
    plt.show()

correlation_plot(result, 'Density_NFO', 'Density_MOL', '20260916_CorrPlots_Labeled.svg', color='tab:blue', fontsize=12, label_col='acronym', labels_to_show=['LH', 'ACAv1', 'LA'])
    
#%%

import scipy
two_cols = result[['Density_NFO', 'Density_MOL']]
two_cols = two_cols.dropna()
regression = scipy.stats.linregress(two_cols['Density_NFO'], two_cols['Density_MOL']) 



simplified_df = result[['Density_NFO', 'Density_MOL', 'acronym', 'names_y']]


import pickle as pkl
with open('20260409_correlation_NFO_P30_MOL_P240', 'wb') as file:
    pkl.dump(simplified_df, file)

simplified_df.to_csv('20260409_correlation_NFO_P30_MOL_P240.csv', index=False)
