#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:03:32 2023

@author: user
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
        

#%%% Experiment name
plot_all = False
import pickle as pkl


fontsize = 14

fig_stand = np.asarray([4.5, 5])
fig_long = np.asarray([4.8,4])

P60_x_lim = 18000

df_concat = open('./NFO_Dataframe.pkl', 'rb')
df_concat = pkl.load(df_concat)

unique_values_list = list(df_concat['exp'].unique())




ish_energy = open('./SagittalGeneExpressionenergyByAllenRegion_regionMetadata.pkl', 'rb')
ish_energy = pkl.load(ish_energy)

ish_density = open('./SagittalGeneExpressionDensityByAllenRegion_regionMetadata.pkl', 'rb')
ish_density = pkl.load(ish_density)

conversion_pickle = open('./AllenISH_Sagittal_GeneInfo.pkl', 'rb')
conversion = pkl.load(conversion_pickle)





exp_name = 'pooled ISH'



### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
pooled_df = df_concat[df_concat['exp'].isin(['LncOL1_P60'])]
df_means = pooled_df.groupby(pooled_df['acronym']).mean(numeric_only=True)

df_non_numeric = pooled_df.groupby(pooled_df['acronym']).first()
df_numeric_means = pooled_df.groupby(pooled_df['acronym']).mean(numeric_only=True)
df_means = df_numeric_means.join(df_non_numeric.select_dtypes(exclude='number'))

#df_means['acronym'] = df_means.index
#df_means.index = df_means['Unamed: 0']

#%% Subregion-specific correlations
isocortex_idx = get_sub_regions_atlas(df_means, child_id=[], sub_keys=[], reg_name = 'Isocortex')
isocortex_atlas_ids = df_means.iloc[isocortex_idx]



df_means = isocortex_atlas_ids
# run corr analysis



#%% First clean up ish data and make sure no duplicate header names
string_cols = ish_energy[['atlas_id', 'acronym', 'id', 'safe_name', 'ontology_id', 'index']]

numeric_cols = ish_energy.drop(columns=['atlas_id', 'acronym', 'id', 'safe_name', 'ontology_id', 'index'])

# Step 2: Average duplicate numeric columns
numeric_avg = numeric_cols.groupby(axis=1, level=0).mean()

# Step 3: Combine back with the string columns
df_result = pd.concat([string_cols, numeric_avg], axis=1)


### Also drop duplicates from conversion table    
conversion = conversion.drop_duplicates(subset='entrez')
conversion = conversion.drop_duplicates(subset='gene_name')




#%% Concatenate both dataframes (ISH and OL density)
# Step 1: Convert both to DataFrames with the value as index


df1_indexed = df_result.set_index('id')
df2_indexed = df_means.set_index('ids')
#df1_indexed = df1_indexed.loc[df2_indexed.index]

common_idx = df1_indexed.index.intersection(df2_indexed.index)
df1_indexed = df1_indexed.loc[common_idx]

common_idx = df2_indexed.index.intersection(df1_indexed.index)
df2_indexed = df2_indexed.loc[common_idx]



# Step 2: Outer join on the index (the integer value)
result = pd.merge(df1_indexed, df2_indexed, left_index=True, right_index=True, how='outer')

# Step 3: Reset index if you want the integer back as a column
result = result.reset_index()

result = result.dropna(axis=1, how='all')  ### drop all nan columns




#%% Compute correlations rapidly
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def corr_analysis(df, target_col):
    # Assume 'cell_density' is the target column
    target = df[target_col]
    # X = df.drop(columns='cell_density')
    
    X = df[[col for col in df.columns if not isinstance(col, str)]]  ### Drop all columns that are string, leaving only genes
    
    genes = X.columns
    
    
    correlations = []
    p_values = []
    
    for i, gene in enumerate(genes):
        x = X[gene]
        valid = target.notna() & x.notna()
        if valid.sum() > 2:
            x_valid = x[valid]
            y_valid = target[valid]
            
            # Center and normalize
            x_c = x_valid - x_valid.mean()
            y_c = y_valid - y_valid.mean()
            
            r = np.dot(x_c, y_c) / ((len(x_c) - 1) * x_c.std(ddof=1) * y_c.std(ddof=1))
            r = np.clip(r, -1, 1)
    
            # p-value
            t_stat = r * np.sqrt((len(x_c) - 2) / (1 - r**2))
            p = 2 * t.sf(np.abs(t_stat), df=len(x_c) - 2)
    
            correlations.append(r)
            p_values.append(p)
        else:
            correlations.append(np.nan)
            p_values.append(np.nan)
    
    # Build result
    cor_df = pd.DataFrame({
        'gene': genes,
        'correlation': correlations,
        'p_value': p_values
    })
    
    # Adjust p-values
    cor_df['p_adj'] = multipletests(cor_df['p_value'].fillna(1), method='fdr_bh')[1]
    
    
    return cor_df



#%% Also concatenate gene name
cor_df.rename(columns={'gene': 'entrez'}, inplace=True)
df_merged = cor_df.merge(conversion, on='entrez', how='left')  ### left keeps all values from result df
df_merged = df_merged.sort_values(by='correlation', ascending=False)

# because conversion was altered earlier to look for duplicates, remove any misaligned (gene_name = nan) entries
df_merged = df_merged.dropna(subset=['gene_name'])

import pickle as pkl
with open('20260102_correlation_ISH_energy_NFO_Density_Isocx', 'wb') as file:
    pkl.dump(df_merged, file)

df_merged.to_csv('20260102_correlation_ISH_energy_NFO_Density_Isocx.csv', index=False)



