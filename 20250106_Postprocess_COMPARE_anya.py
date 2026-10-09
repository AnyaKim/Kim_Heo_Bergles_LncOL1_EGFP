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


from postprocess_func import *
import seaborn as sns
from scipy import stats   
from adjustText import adjust_text

import sys
sys.path.append("..")

from get_brain_metadata import *

from PLOT_FUNCTIONS_postprocess_compare import *
        

#%% List of brains

new_large_OL = 0

#list_brains = get_metadata(mouse_num = 'all')
list_brains = get_metadata(mouse_num = 'all')
#list_brains = get_metadata(mouse_num = ['M246','M248','AK11','AK13','AK14','AK15','AK16','AK18','AK19','AK20','AK21','AK22','AK23','AK24'])

# list_brains = get_metadata(mouse_num = ['M246'])
if new_large_OL:
    list_brains = get_metadata(mouse_num = ['M260', 'M286'])

# print('CUPRIZONE AND OLD BRAIN (anything with lots of lipofuscin) currently using -10grid (only 1st old brain, all newer analyzed old brains using -15grid)')

sav_fold = '/media/user/20TB_HDD_Anya/20251230_plots/'


# myelin_path = glob.glob(os.path.join(downsampled_dir,'*_ch0_PAD.tif'))[0]    # can switch this to "*truth.tif" if there is no name for "input"
# auto_path = glob.glob(os.path.join(downsampled_dir, '*_ch1_PAD.tif'))[0]

pad = True

XY_res = 1.152035240378141
Z_res = 5
res_diff = XY_res/Z_res

ANTS = 1

#%%%% Parse the json file so we can choose what we want to extract or mask out
with open('../atlas_ids/atlas_ids.json') as json_file:
    data = json.load(json_file)
 
     
data = data['msg'][0]

keys_tmp = get_ids_all(data, all_keys=[], keywords=[''])  
keys_tmp = pd.DataFrame.from_dict(keys_tmp)


#%%%% Parse pickles from all folders
""" Loop through all the folders and pool the dataframes!!!"""

# all_coords_df = []
all_keys_df = []
for id_f, info in enumerate(list_brains):

    #fold = info['path'] + info['name'] + '_postprocess'
    fold = info['path'] + info['name'] + '_postprocess_60' #adding this to analyze counts from updated 60 threshold

    pkl_to_use = info['pkl_to_use']

    if not ANTS:
        keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid.pkl'))    
    else:
        # keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY.pkl'))   
        
        if pkl_to_use == 'MYELIN':
            keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE.pkl'))   
            
 
        elif pkl_to_use == 'CUBIC':
            keys = glob.glob(os.path.join(fold, '*_keys_df_ALLEN_EVERYTHING-10grid_ANTS_MY_SIZE_CUBIC.pkl')) 
            
            
    try:    
        keys_df = pd.read_pickle(keys[0])
    except:
        print('Missing: ' + fold)
        continue
    # coords_df = pd.read_pickle(coords[0])
    
    # print('number of cells: ' + str(len(coords_df)))
    
    keys_df['acronym'] = keys_tmp['acronym']
    keys_df['dataset'] = info['num']
    keys_df['exp'] = info['exp']
    keys_df['sex'] = info['sex']
    keys_df['age'] = info['age']   
    
    
    
    drop_ids = []
    #%%% REMOVE ADDITIONAL TORN TISSUE REGIONS - as specified in metadata (including replacements if given hemisphere)
    
    if 'exclude' in info:
        print('removing torn regions')
        # zzz
        
        exclude = info['exclude']
        

        ### ADDITIONAL AREAS TO EXCLUDE FOR ALL BRAINS        
        additional_exclude = [
            ['ME', 'B'],   
            
            ['RCH', 'B'],   ### Hypothalamaic
            ['VMH', 'B'],
            ['Xi', 'B'],    ### Thalamus
            
            ['SCO', 'B'],   ### Midbrain
            ['LING', 'B'],  ### Cerebellum
            
            ]
        
        exclude = exclude + additional_exclude
        
        
        ### do the entire brain
        if exclude[0][0] == 'all':
                keys_df['num_OLs_L'] = keys_df['num_OLs_' + exclude[0][1]]
                keys_df['num_OLs_R'] = keys_df['num_OLs_' + exclude[0][1]]
                keys_df['num_OLs_W'] = keys_df['num_OLs_' + exclude[0][1]] * 2  
        
        
        ### Do by region
        else:
            all_ids = []
            for region in exclude:
                
                
                reg_ids = get_sub_regions_by_acronym(keys_df, child_id=[], sub_keys=[], reg_name=region[0])
                
                ### if we decide to only numbers from one hemisphere, then overwrite data of other hemisphere
                if region[1] != 'B' and region[1] != '':
                    
                    keys_df.loc[reg_ids, 'num_OLs_L'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]]
                    keys_df.loc[reg_ids, 'num_OLs_R'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]]
                    keys_df.loc[reg_ids, 'num_OLs_W'] = keys_df.loc[reg_ids, 'num_OLs_' + region[1]] * 2
        
                else:
                
                    drop_ids = np.concatenate((drop_ids, reg_ids))
                
                

                
        

    #%%%% REMOVE HINDBRAIN ENTIRELY
    
    hindbrain_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Hindbrain')
    drop_ids = np.concatenate((drop_ids, hindbrain_ids))
        
    
    #% ALSO REMOVE Midbrain, behavioral state related (deep structures)
    
    mid_ids = get_sub_regions_atlas(keys_df, child_id=[], sub_keys=[], reg_name='Midbrain, behavioral state related')
    drop_ids = np.concatenate((drop_ids, mid_ids))
        
     
    ### Actually drop here
    children_to_erase = keys_df.iloc[drop_ids]['ids'].values
    
    
    
    
    keys_df = keys_df.drop(drop_ids)#.reset_index()

    new_childs = []
    for id_it, row in keys_df.iterrows():
        
        common = set(children_to_erase) & set(row['children'])
        a = [i for i in row['children'] if i not in common]
    
        new_childs.append(a)
    
    keys_df = keys_df.drop(columns=['children'])
    keys_df['children'] = new_childs  

    ### drop the redundant index columns
    # keys_df = keys_df.drop(columns=['level_0', 'index'])
    



    
    #%%%% sort by depth level and pool all values from children
    
    df_level = keys_df.sort_values(by=['st_level'], ascending=False, ignore_index=False)
    
    for i, row in df_level.iterrows():
        
        childs = row['children']
        
        if len(childs) > 0 and np.isnan(row['atlas_vol_W']):  ### if current row is NAN, then want to pool children, otherwise no
                    
            df_childs = pd.DataFrame()
            for child in childs:
                
                if len(np.where(df_level['ids'] == child)[0]) == 0:  ### means layer 6 was already deleted
                    continue
                id_c = np.where(df_level['ids'] == child)[0][0]
                
                child_row = df_level.iloc[id_c]
                
                child_df = pd.DataFrame([child_row])
                
                df_childs = pd.concat([df_childs, child_df], axis=0)
                
            df_sum = df_childs.sum(axis=0, numeric_only=True)
            
            df_sum = df_sum.drop(['ids', 'parent', 'st_level', 'density_W', 'age'])
            
            row[df_sum.index] = df_sum
            
            
            df_level.loc[i] = row  ### add row back in
          
    keys_df = df_level
    
    if 'side' in info:
        
        keys_df['density_W'] = keys_df['num_OLs_' + info['side']]/keys_df['atlas_vol_' + info['side']]
        keys_df['density_LARGE_W'] = keys_df['num_large_' + info['side']]/keys_df['atlas_vol_' + info['side']] 
        
    else:
        keys_df['density_L'] = keys_df['num_OLs_L']/keys_df['atlas_vol_L']
        keys_df['density_R'] = keys_df['num_OLs_R']/keys_df['atlas_vol_R']
        keys_df['density_W'] = keys_df['num_OLs_W']/keys_df['atlas_vol_W']
        

        if 'num_large_W_CLEAN' in keys_df.keys():
            keys_df['density_LARGE_W_CLEAN'] = keys_df['num_large_W_CLEAN']/keys_df['atlas_vol_W']
        else:
            keys_df['density_LARGE_W'] = keys_df['num_large_W']/keys_df['atlas_vol_W']



    # all_coords_df.append(coords_df)
    all_keys_df.append(keys_df)


df_concat = pd.concat(all_keys_df)



#%%% Experiment name


if new_large_OL:
    exp_name = 'LARGE'
    
else:
    exp_name = 'pooled P60'
    # exp_name = 'Aging'

plot_all = False


fontsize = 14

fig_stand = np.asarray([4.5, 5])
fig_long = np.asarray([4.8,4])

P60_x_lim = 18000






# #%% P60 pooling
# if exp_name == 'pooled P60' or plot_all:

#     exp_name = 'pooled P60'
    
#     ### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
#     pool_all_df = df_concat[df_concat['exp'].isin(['LncOL1_P60'])]
#     df_means = pool_all_df.groupby(pool_all_df.index).mean(numeric_only=True)
#     df_means['acronym'] = keys_df['acronym']
#     df_means['dataset'] = keys_df['dataset']
#     df_means['names'] = keys_df['names']
#     df_means['children'] = keys_df['children']
    
    

#     #%%% Variance plot across all layers sorted by cortex, midbrain, ect...
    
#     ## MAKE SURE TO TURN OFF DROP_ID FOR THIS!!! WANT TO KEEP REGIONS WITH HIGH VARIANCE TO SHOW IT
    
#     areas_to_graph = ['Isocortex', 'Hippocampal formation', 'Hypothalamus', 'Thalamus', 'Midbrain', 'Cerebellum']
#     # to_remove_CORTEX = 'MO|SS|VIS|GU|AUD|VISpl|'
#     to_remove_CORTEX = ''
#     #palette = sns.color_palette("husl")
#     palette = sns.color_palette("Set2")
#     plt.figure(figsize=(3.6, 3))
#     all_areas = []
#     for area in areas_to_graph:
#         plot_vals, names_to_plot = get_subkeys_to_plot(df_means, pool_all_df, reg_name=area, dname='density_W', 
#                                                        to_remove=to_remove_CORTEX, lvl_low=5, lvl_high=9)
#         mean = plot_vals.groupby(['acronym', 'names']).mean(numeric_only=True).reset_index()
#         std = plot_vals.groupby(['acronym', 'names']).std(numeric_only=True).reset_index()
        
#         mean['std'] = std['density_W']
#         mean['mean'] = mean['density_W']
#         mean['cv'] = std['density_W']/mean['density_W']   ### also calculate normalized std (coefficient of variation)
    
#         ### combine mean and variance into same dataframe
#         combined = mean.sort_values(by='cv', ascending=False).reset_index()
        
#         # sns.barplot(std, y=std.index, x='density_W', order=std['acronym'], color='grey',
#         #             errorbar=None)
#         combined['Parcellation'] = area
        
#         all_areas.append(combined)
        
#     all_areas = pd.concat(all_areas).reset_index()
    
#     # get means
#     print(all_areas.groupby('Parcellation')['cv'].mean())
#     print(all_areas.groupby('Parcellation')['cv'].sem())
        
#     ax = sns.barplot(x=all_areas["cv"], y=all_areas["acronym"], orient="h", hue=all_areas['Parcellation'],
#                      palette=palette)
    

#     outlier_ids = np.where(all_areas["cv"] > 0.3)[0]
#     outliers = all_areas.iloc[outlier_ids]
    
#     all_texts = []
#     for idx, row in outliers.iterrows():
#         all_texts.append(plt.text(row['cv'] + 0.01, idx, 
#                   row['acronym'], ha='center', va='center',
#                           size=10, color='black', weight='normal'))


#     sns.move_legend(ax, loc='lower right', frameon=False, title='', fontsize=fontsize)
#     plt.yticks(fontsize=fontsize - 2)
#     plt.yticks([])
#     plt.xticks(fontsize=fontsize - 2)
#     plt.xlim([0, 1.0])
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.ylabel('Brain regions', fontsize=fontsize)
#     plt.xlabel('Coefficient of variation', fontsize=fontsize)
#     plt.tight_layout()

#     adjust_text(all_texts, objects=ax.containers[1], only_move='x+',
#                 arrowprops=dict(arrowstyle='->', color='red'))
    


#     plt.savefig(sav_fold + exp_name +'_all_brain_regions_VARIABILITY.png', format='png', dpi=300)
#     plt.savefig(sav_fold + exp_name +'_all_brain_regions_VARIABILITY.svg', format='svg', dpi=300)
    
    
    
    

#     #%%% Compare across hemispheres
#     # pooled_df = df_concat[df_concat['dataset'].isin(['M229', 'M115', 'M223', 'M126'])]
    
#     # pooled_df = df_concat
    
#     pooled_df = pool_all_df
    
#     df_means = pooled_df.groupby(pooled_df.index).mean(numeric_only=True)
#     df_means['acronym'] = keys_df['acronym']
#     df_means['dataset'] = keys_df['dataset']
#     df_means['names'] = keys_df['names']
#     df_means['children'] = keys_df['children']
        
#     mean_hemisphere = pooled_df.groupby(['acronym', 'names']).mean(numeric_only=True)
    
#     plt.figure(figsize=(3.5, 3.5))
#     ax = plt.gca()
#     pl = sns.regplot(x='density_R', y='density_L', data=mean_hemisphere.dropna(), scatter_kws={"color": "grey", 's':10}, line_kws={"color": "red", "alpha":0.2}, ax=ax)
    
#     #calculate slope and intercept of regression equation
#     slope, intercept, r, p, sterr = stats.linregress(x=pl.get_lines()[0].get_xdata(),
#                                                            y=pl.get_lines()[0].get_ydata())
    
#     #display slope and intercept of regression equation
#     print(slope)


#     r,p = stats.pearsonr(mean_hemisphere.dropna()['density_R'], mean_hemisphere.dropna()['density_L'])
#     print(r)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
    
#     plt.yticks(fontsize=fontsize - 2)
#     plt.xticks(fontsize=fontsize - 2)

#     plt.xlim([0, 80000])
#     plt.ylim([0, 80000])
#     ax.ticklabel_format(axis='x', scilimits=[-3, 3])  ### set to be order of magnitude
#     ax.ticklabel_format(axis='y', scilimits=[-3, 3])
    
    
#     ### Find extreme outliers
#     # check_diff = (np.abs(mean_hemisphere['density_L'] - mean_hemisphere['density_R']))/((np.abs(mean_hemisphere['density_L'] + mean_hemisphere['density_R'])/2))
#     # outlier_idx = np.where(check_diff > 0.4)[0]
#     # outliers = mean_hemisphere.iloc[outlier_idx].reset_index()
    
#     # all_texts = []
#     # for idx, row in outliers.iterrows():
#     #     all_texts.append(plt.text(row['density_L'] +0.01, row['density_R'], 
#     #              row['acronym'], ha='center', va='center',
#     #                       size=10, color='black', weight='normal'))
#     # adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))


#     ### Force calculate p value
#     from mpmath import mp
#     # mp.dps = 1000
    
#     r = mp.mpf(r)
#     n = len(mean_hemisphere.dropna())
    
#     x = (-abs(r) + 1)/2  # shift per `loc=-1`, scale per `scale=2`
#     p = 2*mp.betainc(n/2 - 1, n/2 - 1, 0, x, regularized=True)
#     print(p)

        
#     plt.xlabel('Density right (cells/mm\u00b3)', fontsize=fontsize)
#     plt.ylabel('Density left (cells/mm\u00b3)', fontsize=fontsize)
#     plt.tight_layout()
    
    
    
    
#     plt.savefig(sav_fold + exp_name + '_LvsR_correlation.png', format='png', dpi=300)
#     plt.savefig(sav_fold + exp_name + '_LvsR_correlation.svg', format='svg', dpi=300)

#     ### Find regions that do NOT match
#     # mean_hemisphere['num_OLs_absdiff'] = abs(mean_hemisphere['num_OLs_L'] - mean_hemisphere['num_OLs_R'])
#     # mean_hemisphere['num_OLs_scaleddiff'] = mean_hemisphere['num_OLs_absdiff']/mean_hemisphere['num_OLs_W']   
    
    
    
    
    
    
    
#     #%%% Plot by layer
#     plt.figure(figsize=(3.6, 3))
#     names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
#     #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
#     palette = sns.color_palette("husl", len(names_to_plot))
#     styles = ['-', '--', '-.', ':']
    
#     for i_n, name in enumerate(names_to_plot):
#         # if name matches the first half at minimum, then go and plot
#         match = aging_anya[aging_anya['acronym'].str.contains(name) == True]
#         if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
#             match = match[match['acronym'].str.contains('RSPd4') == False]   
        
#         layers = ['1', '2/3', '4', '5', '6']
        
#         all_layers = []
#         for layer in layers:
        
#             lay_df = match[match['names'].str.contains(layer) == True]
        
#             if len(lay_df) == 0:
#                 continue
        
#             sum_df = lay_df.groupby(['dataset']).sum()
        
#             sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
#             sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
#             sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
#             sum_df['layer'] = layer
            
#             all_layers.append(sum_df)
            
#         df_layers = pd.concat(all_layers)
            
#         ### SKIP if wasnt subdivided into smaller layer units
#         if df_layers['density_W'].isna().any():
#             continue
            
#         # plt.figure()
#         # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
#         sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
#                      errorbar=('se'))
        
    
    
#     ax = plt.gca()
#     plt.yticks(fontsize=fontsize - 2)
#     plt.xticks(fontsize=fontsize - 2)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     #ax.spines['left'].set_visible(False)
#     plt.legend(loc = 'lower right', frameon=False)
    
    
#     plt.xlabel('Cortical layer', fontsize=fontsize)
#     plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
#     ax.set_yscale('log')
#     plt.ylim([0, 100000])
#     plt.tight_layout()
    
#     plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new.png', format='png', dpi=300)
#     plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new.svg', format='svg', dpi=300)


#     # ### Then plot it linear
#     plt.legend(loc = 'upper left', frameon=False)
#     ax.set_yscale('linear')
#     plt.ylim([0, 30000])  ### this ruins log plot...
#     ax = plt.gca()
    
        
#     ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
#     plt.tight_layout()
#     plt.savefig(sav_fold + exp_name + '_by_LAYERS.png', format='png', dpi=300)
#     plt.savefig(sav_fold + exp_name + '_by_LAYERS.svg', format='svg', dpi=300)


    #%%% Plot by layer- For revisions- Panel I 

    #%%%% Plot by layer- Just P30- REVISIONS
    plt.figure(figsize=(3, 3))
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = aging_anya_p30[aging_anya_p30['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
            sum_df['layer'] = layer
            
            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
        # plt.figure()
        # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
        sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                    errorbar=('se'))
    
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    # plt.legend(loc = 'lower right', frameon=False)
    
    
    # plt.xlabel('Cortical layer', fontsize=fontsize)
    # plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
    # ax.set_yscale('log')
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.svg', format='svg', dpi=300)
    
    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    plt.ylim([0, 1750])  ### this ruins log plot...
    ax = plt.gca()
    
        
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P30_only.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P30_only.svg', format='svg', dpi=300)
    
    #%%%% Plot by layer- Just P60- REVISIONS
    plt.figure(figsize=(3, 3))
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = aging_anya_p60[aging_anya_p60['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
            sum_df['layer'] = layer
            
            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
        # plt.figure()
        # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
        sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                    errorbar=('se'))
    
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    # plt.legend(loc = 'lower right', frameon=False)
    
    
    # plt.xlabel('Cortical layer', fontsize=fontsize)
    # plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
    # ax.set_yscale('log')
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.svg', format='svg', dpi=300)
    
    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    plt.ylim([0, 1750])  ### this ruins log plot...
    ax = plt.gca()
    
        
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P60_only.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P60_only.svg', format='svg', dpi=300)
    #%%%% Plot by layer- Just P30 and P60- REVISIONS- Not used

    plt.figure(figsize=(3, 3))
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = aging_anya_p30_60[aging_anya_p30_60['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
            sum_df['layer'] = layer
            
            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
        # plt.figure()
        # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
        sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                    errorbar=('se'))
    
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    # plt.legend(loc = 'lower right', frameon=False)
    
    
    # plt.xlabel('Cortical layer', fontsize=fontsize)
    # plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
    # ax.set_yscale('log')
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.svg', format='svg', dpi=300)
    
    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    plt.ylim([0, 1500])  ### this ruins log plot...
    ax = plt.gca()
    
        
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P30_60.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P30_60.svg', format='svg', dpi=300)
    
    #%%%% Plot by layer- Just P140
    plt.figure(figsize=(3, 3))
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = aging_anya_p140[aging_anya_p140['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
            sum_df['layer'] = layer
            
            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
        # plt.figure()
        # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
        sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                    errorbar=('se'))
    
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    # plt.legend(loc = 'lower right', frameon=False)
    
    
    # plt.xlabel('Cortical layer', fontsize=fontsize)
    # plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
    # ax.set_yscale('log')
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.svg', format='svg', dpi=300)
    
    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    plt.ylim([0, 1750])  ### this ruins log plot...
    ax = plt.gca()
    
        
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P140_only.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P140_only.svg', format='svg', dpi=300)
    
    #%%%% Plot by layer- Just P500
    plt.figure(figsize=(3, 3))
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    #names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    for i_n, name in enumerate(names_to_plot):
        # if name matches the first half at minimum, then go and plot
        match = aging_anya_p500[aging_anya_p500['acronym'].str.contains(name) == True]
        if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
            match = match[match['acronym'].str.contains('RSPd4') == False]   
        
        layers = ['1', '2/3', '4', '5', '6']
        
        all_layers = []
        for layer in layers:
        
            lay_df = match[match['names'].str.contains(layer) == True]
        
            if len(lay_df) == 0:
                continue
        
            sum_df = lay_df.groupby(['dataset']).sum()
        
            sum_df['density_L'] = sum_df['num_OLs_L']/sum_df['atlas_vol_L']
            sum_df['density_R'] = sum_df['num_OLs_R']/sum_df['atlas_vol_R']
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
        
            sum_df['layer'] = layer
            
            all_layers.append(sum_df)
            
        df_layers = pd.concat(all_layers)
            
        ### SKIP if wasnt subdivided into smaller layer units
        if df_layers['density_W'].isna().any():
            continue
            
        # plt.figure()
        # sns.boxplot(x=df_layers['layer'], y=df_layers['density_W'])
        sns.lineplot(x=df_layers['layer'], y=df_layers['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                    errorbar=('se'))
    
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    # plt.legend(loc = 'lower right', frameon=False)
    
    
    # plt.xlabel('Cortical layer', fontsize=fontsize)
    # plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    
    
    # ax.set_yscale('log')
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_by_LAYERS_LOG_new_P30_60.svg', format='svg', dpi=300)
    
    # ### Then plot it linear
    plt.legend(loc = 'upper left', frameon=False)
    ax.set_yscale('linear')
    plt.ylim([0, 1750])  ### this ruins log plot...
    ax = plt.gca()
    
        
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    
    
    
    plt.tight_layout()
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P500_only.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_by_LAYERS_linear_new_P500_only.svg', format='svg', dpi=300)
    
    #%% For revisions:
    # 1 --- highlight L6-MOs and L6-PL have higher slope than all other cortical regions
    
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Cortical regions and layers of interest
    cortical_names = ['SSp', 'MOp', 'RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'PERI', 'TEa', 'PL', 'ACA', 'AI',
                      
                      'GU', 'VISC', 'ILA', 'ORB', 'PTLp', 'ECT'
                      
                      ]
    layers = ['1', '2/3', '4', '5', '6']
    
    # Filter for P240–P850
    late_aging_df = aging_df[(aging_df['age'] >= 240) & (aging_df['age'] <= 850)]
    
    # Initialize results list
    results = []
    
    # Loop over cortical regions and layers
    for region in cortical_names:
        region_df = late_aging_df[late_aging_df['acronym'].str.contains(region)]
        
        # Optional: exclude RSPd4 if needed
        if region == 'RSP':
            region_df = region_df[~region_df['acronym'].str.contains('RSPd4')]
        
        for layer in layers:
            layer_df = region_df[region_df['names'].str.contains(layer)]
            
            if layer_df.empty:
                continue
            
            # Average across datasets for each age
            grouped_df = layer_df.groupby('age').mean(numeric_only=True).reset_index()
            if len(grouped_df) >= 2:
                slope, intercept, r, p, stderr = stats.linregress(grouped_df['age'], grouped_df['density_W'])
                # Fold change P240->P850
                P240 = grouped_df[grouped_df['age'] == 240]['density_W'].mean()
                P850 = grouped_df[grouped_df['age'] == 850]['density_W'].mean()
                fold_change = P850 / P240 if P240 > 0 else np.nan
                
                results.append({
                    'region': region,
                    'layer': layer,
                    'slope': slope,
                    'fold_change': fold_change
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Define "late risers" (L6-MOs, L6-PL)
    results_df['group'] = 'Others'
    results_df.loc[(results_df['region'] == 'MOs') & (results_df['layer'] == '6'), 'group'] = 'L6-MOs'
    results_df.loc[(results_df['region'] == 'PL') & (results_df['layer'] == '6'), 'group'] = 'L6-PL'
    late_risers_df = results_df[results_df['group'].isin(['L6-MOs', 'L6-PL'])]
    other_regions_df = results_df[results_df['group'] == 'Others']
    
    # Compare slopes
    t_slope, p_slope = stats.ttest_ind(late_risers_df['slope'], other_regions_df['slope'], equal_var=False)
    print(f"Slope comparison (late risers vs others): t={t_slope:.3f}, p={p_slope:.4f}")
    
    # Compare fold-change
    t_fold, p_fold = stats.ttest_ind(late_risers_df['fold_change'], other_regions_df['fold_change'], equal_var=False)
    print(f"Fold-change comparison (late risers vs others): t={t_fold:.3f}, p={p_fold:.4f}")
    
    # Save results
    results_df.to_csv(sav_fold + exp_name + '_cortical_layers_late_aging.csv', index=False)
    
    # Plot slopes
    # Plot slopes without boxplot outliers
    from adjustText import adjust_text
    
    plt.figure(figsize=(2.5, 3.5))  # Tall and thin figure
    
    # Make a thin stripplot (minimal x jitter)
    ax = sns.stripplot(
        data=results_df,
        x=[''] * len(results_df),  # no x-axis label
        y='slope',
        color='gray',
        jitter=0.05,  # subtle horizontal spread
        size=6,
        alpha=0.7
    )
    
    # Compute median and IQR
    median_slope = results_df['slope'].median()
    q1 = results_df['slope'].quantile(0.25)  # 25th percentile
    q3 = results_df['slope'].quantile(0.75)  # 75th percentile
    
    # Plot median as a solid line
    plt.axhline(median_slope, color='red', linewidth=2, label='Median')
    
    # Plot IQR as dashed lines
    plt.axhline(q1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='IQR')
    plt.axhline(q3, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Annotate top 5 regions/layers
    top5_df = results_df.sort_values(by='slope', ascending=False).head(5)
    texts = []
    for idx, row in top5_df.iterrows():
        texts.append(
            plt.text(
                x=0.02, y=row['slope'],  # small x offset
                s=f"{row['region']} L{row['layer']}",
                fontsize=10,
                ha='left',
                va='center'
            )
        )
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    # Clean up aesthetics
    plt.ylabel('Slope of density (P240–P850)', fontsize=12)
    plt.xticks([])  # remove x-axis ticks
    plt.yticks(fontsize=10)
    plt.ylim([-2, 40])  # Start slightly below 0
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(sav_fold + exp_name + '_clean_stripplot_top5_median_iqr.svg', dpi=300)
    plt.show()
    
    # Print Top 5 Summary
    print("\nTop 5 cortical regions/layers by late-life slope (P240–P850):")
    for idx, row in top5_df.iterrows():
        print(f"{row['region']} Layer {row['layer']}: slope = {row['slope']:.4f}, fold-change = {row['fold_change']:.2f}")
    
    
    # Compute mean and SEM for slope and fold-change
    mean_slope = results_df['slope'].mean()
    sem_slope = stats.sem(results_df['slope'], nan_policy='omit')
    
    mean_fold_change = results_df['fold_change'].mean()
    sem_fold_change = stats.sem(results_df['fold_change'], nan_policy='omit')
    
    print(f"\nMean slope across all cortical regions/layers: {mean_slope:.4f} ± {sem_slope:.4f}")
    print(f"Mean fold-change across all cortical regions/layers: {mean_fold_change:.2f} ± {sem_fold_change:.2f}")

    
    
    
    # #%%% PLOT global regions of interest
    #  #Anya's version below

    # names_to_plot = [
    #                   'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
    #                  'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
    #                  'AUD', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
    #                  'VIS',                                             # occipital lobe
    #                  # 'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 'RHP', 'ENT',   ### Hippocampus

    #                  ]
    # palette = sns.color_palette("Set2", len(names_to_plot))

        
    # plot_global_LR(pooled_df, names_to_plot, [palette[0], palette[3]], sav_fold, exp_name + '_GLOBAL_COMPARISON_low',
    #                ylim=15000, figsize=(4.4, 3))

    # names_to_plot = [
    #                 'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg', 'RHP', 'ENT',   ### Hippocampus
    #                  'MOB', 'PIR', 'COA', 'PAA', 'NLOT', 'TR',  ### Olfactory areas, Pririform-Amygdalar, Cortical-amygdalar, Postpiriform Transition area (TR), Nuclus of olfactory Tract (NLOT)
                     
    #                  'PAL', 'PALv', 'PALm', 'PALc', 'GPe', 'GPi',   ### Pallidum
    #                  'STR', 'CP', 'ACB', 'LSX', 'sAMY',   ### ACB --- nucleus acumbens, Striatum-like amygdalar nuclei (SAMY)
    #                  'TH','DORpm', 'DORsm', 'VAL', 'VM', 'MG', 'LGd',
    #                  'HY',    # Hypothalamus, contains ME (Medial eminence)
    #                  # 'MBsen', 'MBmot', 'MBsta',   ### sensory IC/SC, motor, behavioral
    #                  'SCs', 'SCm',  'IC', 'SNr', 'VTA', 'PAG',    ###'SNc', SC superior colliculus motor (m) or sensory (s) or compact (c), PAG periaqueductal gray
                     
                     
    #                  # 'P',   # Pons and Medulla --- currently skipped
    #                  'CB', 'VERM', 'HEM', 'CBN',     # Cebreellum, CBN - cerebellar nuclei
    #                   # 'cc', 'fxs', 'arb'   #'fiber tracts'  arb == arbor vitae in cerebellar related fiber tracts (cbf)
    #                  ]
    # plot_global_LR(pooled_df, names_to_plot, [palette[0], palette[3]], sav_fold, exp_name + '_GLOBAL_COMPARISON_high',
    #                ylim=60000, figsize=(8, 4))


    

    
#     #%% P60 pooling- Not used for revisions
#     if exp_name == 'pooled P60' or plot_all:

#         exp_name = 'pooled P60'
        
#         ### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
#         pool_all_df = df_concat[df_concat['exp'].isin(['LncOL1_P60'])]
#         df_means = pool_all_df.groupby(pool_all_df.index).mean(numeric_only=True)
#         df_means['acronym'] = keys_df['acronym']
#         df_means['dataset'] = keys_df['dataset']
#         df_means['names'] = keys_df['names']
#         df_means['children'] = keys_df['children']
        
        
    
#     names_to_plot = [
#                       'cc', 'ccg', 'ccs', 'fp',   # cc
                      
                      
#                       'scp', 'mcp', 'icp',        # cerebellum

#                       'fxs', 'alv', 'df', 'fi',   # fornix
                      
#                       'SSp'  #

#                      ]
#     palette = sns.color_palette("Set2", len(names_to_plot))

        
#     plot_global(pool_all_df, names_to_plot, palette, sav_fold, exp_name + '_ANYA_WM_P60',
#                    ylim=2000, figsize=(7, 3))
    
    
    
#     if exp_name == 'pooled P100' or plot_all:

#         exp_name = 'pooled P100'
        
#         ### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
#         pool_all_df = df_concat[df_concat['exp'].isin(['LncOL1_P100'])]
#         df_means = pool_all_df.groupby(pool_all_df.index).mean(numeric_only=True)
#         df_means['acronym'] = keys_df['acronym']
#         df_means['dataset'] = keys_df['dataset']
#         df_means['names'] = keys_df['names']
#         df_means['children'] = keys_df['children']
        
        
#  #Getting all the large WM regions   
#     names_to_plot = [
#                       'cc', 'arb', 'mfbs', 'fxs','onl','cbp','int','or','fiber tracts','fi','alv','lfbst','cing','eps','scwm','hc','ec','ll'
#                      ]
#     palette = sns.color_palette("Set2", len(names_to_plot))

        
#     plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_ANYA_WM',
#                    ylim=2000, figsize=(7, 3))
    
# #Looking at just a couple relevant examples
    
#     names_to_plot = [
                      
#                       'arb', 'mfbs','or', #drop consistently over time
#                       'fi', #stays the same
#                       'cc', #stays the same until P500
                     
#                      ]
#     palette = sns.color_palette("Set2", len(names_to_plot))

        
#     plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_ANYA_WM_subset',
#                    ylim=1200, figsize=(7, 3))
        

    
#  #Getting GM regions   
#     names_to_plot = [
#                       'MOp', 'SSp', 'MOs','SSs', 'PTLp','RSP','VISp','VISa','VISl','AUDp',         ### Frontal lobe
#                       'AUDd',  'AUDv',          
#                        'TEa', 'ECT', 'PERI'                                           # occipital lobe
#                      ]
#     palette = sns.color_palette("Set2", len(names_to_plot))

        
#     plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_ANYA_GM_Ordered_all_ages',
#                    ylim=1000, figsize=(9, 3))    
    
    
#     names_to_plot = [
#         'MOp', 'SSp', 'VISrl', 'AUDp',  'VISl', 'MOs', 'SSs',      ### Frontal lobe
#        'VISp', 'AUDd', 'PTLp', 'AUDv', 'VISal', 'AUDpo',          ### Parietal lobe
#        'RSP', 'VISa', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
#        'ILA'                                             # occipital lobe
                     
#                      ]
#     palette = sns.color_palette("Set2", len(names_to_plot))

        
#     plot_global(aging_anya_p30, names_to_plot, palette, sav_fold, exp_name + '_ANYA_GM_Ordered_only_P30',
#                    ylim=1000, figsize=(7, 3))   
    
    
#     if exp_name == 'pooled P500' or plot_all:

#         exp_name = 'pooled P500'
        
#         ### FOR THE FIRST PART OF ANALYSIS USE ALL P60 brains
#         pool_all_df = df_concat[df_concat['exp'].isin(['LncOL1_P500'])]
#         df_means = pool_all_df.groupby(pool_all_df.index).mean(numeric_only=True)
#         df_means['acronym'] = keys_df['acronym']
#         df_means['dataset'] = keys_df['dataset']
#         df_means['names'] = keys_df['names']
#         df_means['children'] = keys_df['children']
        
        
    
#     names_to_plot = [
#                       'cc', 'ccg', 'ccs', 'fp',   # cc
                      
                      
#                       'scp', 'mcp', 'icp',        # cerebellum

#                       'fxs', 'alv', 'df', 'fi',   # fornix
                      
#                       'SSp'  #

#                      ]
#     palette = sns.color_palette("Set2", len(names_to_plot))

        
#     plot_global(pool_all_df, names_to_plot, palette, sav_fold, exp_name + '_ANYA_WM',
#                    ylim=2000, figsize=(7, 3))
    
    
    
    
    
    


#%% AGING pooling- Anya- run this

if exp_name == 'Aging' or plot_all:
    
    exp_name = 'Aging'
    
    ### Exclude M229 --- mostly just for large OL counting since stitching was only translational
    ### Included for now... just for fun
    # aging_df = df_concat[~df_concat['dataset'].isin(['M229'])]
    aging_df = df_concat
    
    # aging_df = aging_df[aging_df['exp'].isin(['P60', 'P120', 'P240', 'P360', 'P620', 'P100', 'P800', 'P60_NEW', 'P60_NEW_NEW', 'P60_NEW_ORIG',
    #                                           'P120_NEW', 'P120_NEW_NEW', 'P240_NEW', 'P240_NEW_NEW','P620_NEW', 'P620_NEW_NEW', 'P800_NEW', 'P800_NEW_NEW'])]
    
    
    aging_anya = aging_df[aging_df['exp'].isin(['LncOL1_P30','LncOL1_P60','LncOL1_P140','LncOL1_P500','LncOL1_P60_b'])]
    aging_anya_p30 = aging_df[aging_df['exp'].isin(['LncOL1_P30'])]
    aging_anya_p60 = aging_df[aging_df['exp'].isin(['LncOL1_P60'])]
    aging_anya_p30_60 = aging_df[aging_df['exp'].isin(['LncOL1_P30','LncOL1_P60',])]
    aging_anya_p140 = aging_df[aging_df['exp'].isin(['LncOL1_P140',])]    
    aging_anya_p500 = aging_df[aging_df['exp'].isin(['LncOL1_P500'])]
    
    df_means = aging_anya.groupby(aging_anya.index).mean(numeric_only=True)
    df_means['acronym'] = keys_df['acronym']
    df_means['dataset'] = keys_df['dataset']
    df_means['names'] = keys_df['names']
    df_means['children'] = keys_df['children']
   
    
 
    # #%%% Compare GLOBAL regions- Line Graph
    
    # #%%%% Compare GLOBAL regions- All regions
    # names_to_plot = ['Isocortex', 
    #                  # 'HPF', 
    #                  'HIP', 'RHP',
    #                  # 'CNU',
    #                  'STR', #'PAL',
    #                  'IB', #'TH', 'HY',   # Thalamus/Hypothalamus
    #                  'CBX', ### exclude cerebellar nuclei and arbor vitae later?
    #                  'fiber tracts', 
    #                  #'MB'
    #                  ]
    # palette = sns.color_palette("husl", len(names_to_plot))
    
    # palette[-1] =sns.color_palette("Set2", len(names_to_plot))[5]
    
    # compare_GLOBAL_regions(aging_anya, names_to_plot, palette, xlim=[25, 60], ylim=1500, ylim_norm=[1,2.2], sav_name='AGING_anya_new_tp', sav_fold=sav_fold, exp_name=exp_name, 
    #                        fontsize=14, add_zero=False, figsize=(4,3))
    
    # #%%%% Compare GLOBAL regions- Line Graph
    
    
    # #%%%% Compare GLOBAL regions- Fiber tracts only  
    # names_to_plot = [
    #                   'fi', 'cc','hc',      ### Dorsal
    #                  'int', 'amc','cpd','fx','sm', ###Ventral
    #                  'In', 
    #                  'SSp'
                     
    #                  ]
    # palette = sns.color_palette("husl", len(names_to_plot))
    
    # palette[-1] =sns.color_palette("Set2", len(names_to_plot))[5]
    
    # compare_GLOBAL_regions(aging_anya, names_to_plot, palette, xlim=[25, 60], ylim=2000, ylim_norm=[1,2.2], sav_name='AGING_anya-WM_over_time', sav_fold=sav_fold, exp_name=exp_name, 
    #                        fontsize=14, add_zero=False, figsize=(4,3))
    
    
    # names_to_plot = [
    #                   'fi', 'cc','hc'      ### Dorsal
    #                  #'int', 'amc','cpd','fx','sm', ###Ventral
    #                  #'In', 
    #                  'SSp'
                     
    #                  ]
    # palette = sns.color_palette("husl", len(names_to_plot))
    
    # palette[-1] =sns.color_palette("Set2", len(names_to_plot))[5]
    
    # compare_GLOBAL_regions(aging_anya, names_to_plot, palette, xlim=[25, 60], ylim=2000, ylim_norm=[1,2.2], sav_name='AGING_anya-WM_over_time_dorsal_only', sav_fold=sav_fold, exp_name=exp_name, 
    #                        fontsize=14, add_zero=False, figsize=(4,3))
    
    
    # # Compare fiber tracts P60 vs. P640
    # for name in names_to_plot:
    #     match = aging_anya[aging_anya['acronym'].str.fullmatch(name) == True]
    #     P60 = match[match['age'] == 60]
    #     P620 = match[match['age'] == 620]    
    #     print('Change in density: ' + str(np.mean(P620['density_W']) - np.mean(P60['density_W'])) + name)
    
    # # Compare fiber tracts P30 vs. P60- Anya
    # for name in names_to_plot:
    #     match = aging_anya[aging_anya['acronym'].str.fullmatch(name) == True]
    #     P30 = match[match['age'] == 30]
    #     P60 = match[match['age'] == 60]    
    #     print('Change in density: ' + str(np.mean(P30['density_W']) - np.mean(P60['density_W'])) + name)
    
    
    # also compare fold change
    # for name in names_to_plot:
    #     match = aging_df[aging_df['acronym'].str.fullmatch(name) == True]
    #     P60 = match[match['age'] == 60]
    #     P620 = match[match['age'] == 620]    
    #     print('Change in density: ' + str(np.mean(P620['density_W'])/np.mean(P60['density_W'])) + name)
    
                
    
    
    
    # motor_df = match[match['names'].str.contains('2/3') == True]
    # print(stats.sem(P60['density_W']))   
    # print(stats.sem(P620['density_W']))   
    

    # tstat, p = stats.ttest_ind(P60['density_W'], P620['density_W'], equal_var=True, alternative='two-sided')
    # print(p)
    # print('P60:' + str(np.mean(P60['density_W'])) + ' P620: ' + str(np.mean(P620['density_W'])))


    # ### Also compare ratio of primary vs. secondary area
    
    # match = aging_df[aging_df['acronym'].str.fullmatch('SSp') == True]
    # P60_SSp = match[match['age'] == 60]
    # P620_SSp = match[match['age'] == 620]    
    
    # match = aging_df[aging_df['acronym'].str.fullmatch('SSs') == True]
    # P60_SSs = match[match['age'] == 60]
    # P620_SSs = match[match['age'] == 620]    
        
    # print('Primary vs. Secondary at P60: ' + str(np.mean(P60_SSp['density_W'])/np.mean(P60_SSs['density_W'])) + 'SSp P60')
    # print('Primary vs. Secondary at P620: ' + str(np.mean(P620_SSp['density_W'])/np.mean(P620_SSs['density_W'])) + 'SSp P620')

    # match = aging_df[aging_df['acronym'].str.fullmatch('MOp') == True]
    # P60_MOp = match[match['age'] == 60]
    # P620_MOp = match[match['age'] == 620]    
    
    # match = aging_df[aging_df['acronym'].str.fullmatch('MOs') == True]
    # P60_MOs = match[match['age'] == 60]
    # P620_MOs = match[match['age'] == 620]    
        
    # print('Primary vs. Secondary at P60: ' + str(np.mean(P60_MOp['density_W'])/np.mean(P60_MOs['density_W'])) + 'MOp P60')
    # print('Primary vs. Secondary at P620: ' + str(np.mean(P620_MOp['density_W'])/np.mean(P620_MOs['density_W'])) + 'MOs P620')


    
    # ### Compare Hippocampal Dentate Gryus across timepoints
    # DG_po_df = aging_df[aging_df['acronym'].str.fullmatch('DG-po') == True]
    # DG_mo_df = aging_df[aging_df['acronym'].str.fullmatch('DG-mo') == True]
    # DG_sg_df = aging_df[aging_df['acronym'].str.fullmatch('DG-sg') == True]
    
    # ages = [60, 240, 620, 850]
    # for age in ages:
    #     DG_po = DG_po_df[DG_po_df['age'] == age]
    #     DG_mo = DG_mo_df[DG_mo_df['age'] == age]    
    #     DG_sg = DG_sg_df[DG_sg_df['age'] == age]
        
    #     # print(stats.sem(DG_po['density_W']))   
    #     # print(stats.sem(DG_mo['density_W']))  
    #     # print(stats.sem(DG_sg['density_W']))  
    
    #     tstat, p = stats.ttest_ind(DG_po['density_W'], DG_mo['density_W'], equal_var=True, alternative='two-sided')
    #     print(p)
    #     print(str(age) + '  ' + str(np.mean(DG_po['density_W'])) + '   ' + str(np.mean(DG_mo['density_W'])))
        
    #     tstat, p = stats.ttest_ind(DG_po['density_W'], DG_sg['density_W'], equal_var=True, alternative='two-sided')
    #     print(p)
    #     print(str(age) + '  ' + str(np.mean(DG_po['density_W'])) + '   ' + str(np.mean(DG_sg['density_W'])))


    
    



    #%% Get df Layers
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    
    # names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    
    layers = ['1', '2/3', '4', '5', '6']
    
    fontsize = 14
    
    #%%% Compare layers for each cortical region
    all_layers = []
    for layer in layers:
        plt.figure(figsize=fig_long)
        for i_n, name in enumerate(names_to_plot):
            # if name matches the first half at minimum, then go and plot
            match = aging_anya[aging_anya['acronym'].str.contains(name) == True]
            if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
                match = match[match['acronym'].str.contains('RSPd4') == False]   
            match=match.reset_index()
            
    
            ### DROP layer 1 from M127 and M126 for the moment due to delipidation artifiact
            if layer == '1':
                match = match.drop(index=np.where(match['dataset'] == 'M127')[0]).reset_index()        
                match = match.drop(index=np.where(match['dataset'] == 'M126')[0])
    
            lay_df = match[match['names'].str.contains(layer) == True]
 
    #%%% Compare GLOBAL regions
     
            
            sum_df = lay_df.groupby(['dataset', 'exp', 'age']).sum(numeric_only=True)
            sum_df = sum_df.reset_index() ### moves all of the indices into columns!!!
    
            sum_df['density_W'] = sum_df['num_OLs_W']/sum_df['atlas_vol_W']
    
            sns.lineplot(x=sum_df['age'], y=sum_df['density_W'], label=name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)]   )
    
            plt.title('Layer ' + layer)
    
            # give name of layer
            sum_df['layer'] = layer
        
            all_layers.append(sum_df)
            
    df_layers = pd.concat(all_layers)
    
    df_layers = df_layers.groupby(['dataset', 'layer', 'age']).sum(numeric_only=True)     
    df_layers['density_W'] = df_layers['density_W']/df_layers['atlas_vol_W']   ### scale it down since it's sum above   
    
    df_layers = df_layers.reset_index()
    
    #%%% Plot layers over time
    names_to_plot = ['SSp', 'SSs', 'MOp', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'RSP', 'TEa']
    # names_to_plot = ['SSp', 'SSs', 'MOp', 'ORB', 'RSP']
    
    palette = sns.color_palette("husl", len(names_to_plot))
    styles = ['-', '--', '-.', ':']
    
    fontsize = 14
    
    layer_colors = ['deepskyblue',
                    'blueviolet',
                    'forestgreen',
                    'goldenrod',
                    'grey'
                    ]
    layers = ['1', '2/3', '4', '5', '6']
    plt.figure(figsize=(3.5, 3))
    for i_n, layer in enumerate(layers):
        match = df_layers[df_layers['layer'].str.contains(layer) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        
        match['fold_change'] = match['density_W']/baseline
        
        sns.lineplot(x=match['age'], y=match['fold_change'], label=layer, color=layer_colors[i_n % len(layer_colors)], linestyle=styles[i_n % len(styles)],
                      alpha=0.8,
                      errorbar='se')
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([1, 3])
    plt.xlabel('Age (days)', fontsize=fontsize)
    plt.ylabel('Density fold change', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_LAYERS_over_time.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_LAYERS_over_time.svg', format='svg', dpi=300)
    
    ### Plt RAW DENSITY of OLs over time
    layer_colors = ['deepskyblue',
                    'blueviolet',
                    'forestgreen',
                    'goldenrod',
                    'grey'
                    ]
    plt.figure(figsize=(3.5, 3))
    for i_n, layer in enumerate(layers):
        match = df_layers[df_layers['layer'].str.contains(layer) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        
        match['raw_change'] = match['density_W'] - baseline
        
        sns.lineplot(x=match['age'], y=match['raw_change'], label=layer, color=layer_colors[i_n % len(layer_colors)], linestyle=styles[i_n % len(styles)],
                      alpha=0.8,
                      errorbar='se')
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([0, 6000])
    plt.xlabel('Age (days)', fontsize=fontsize)
    plt.ylabel('Density (cells/mm\u00b3)', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_LAYERS_DENSITY_CHANGE_over_time.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_LAYERS_DENSITY_CHANGE_over_time.svg', format='svg', dpi=300)
    
       
    
    #%%% Plot by time for cortical regions
    to_remove = 'MO|SS|AUDp|AUDd|VISpl|VISrl|AUDv|VISal|VISp|VISl|VISa|AUDpo|VISpor|VISam|VISli|VISpm'
    
    to_remove = to_remove + '|ECT|PERI|PL'  ### still some registration issues at 4mos
    
    to_remove = to_remove + '|ORB|GU|ACA|PTLp|Al|FRP|ILA'
    plot_vals, names_to_plot = get_subkeys_to_plot(df_means, aging_df, reg_name='Isocortex', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=9)
    
    all_avg = []
    plt.figure(figsize=(3.5, 3))
    for i_n, reg_name in enumerate(names_to_plot):
        match = plot_vals[plot_vals['acronym'].str.fullmatch(reg_name) == True]
        baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
        
        match['fold_change'] = match['density_W']/baseline
        
        sns.lineplot(x=match['age'], y=match['fold_change'], label=reg_name, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
                     errorbar='se')
        
    ax = plt.gca()
    plt.yticks(fontsize=fontsize - 2)
    plt.xticks(fontsize=fontsize - 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    plt.legend(loc = 'upper left', frameon=False)
    plt.ylim([1, 3])
    plt.xlabel('Age (days postnatal)', fontsize=fontsize)
    plt.ylabel('Density fold change', fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(sav_fold + exp_name + '_CORTICAL_REGIONS_over_time.png', format='png', dpi=300)
    plt.savefig(sav_fold + exp_name + '_CORTICAL_REGIONS_over_time.svg', format='svg', dpi=300)
    
    
    
    
    # #%%% plot scatterplot comparing starting density and final fold change
    # #to_remove = 'MO|SS|AUDp|AUDd|VISpl|VISrl|AUDv|VISal|VISp|VISl|VISa|AUDpo|VISpor|VISam|VISli|VISpm'
    # #to_remove = to_remove + '|ECT|PERI|PL'  ### still some registration issues at 4mos
    # to_remove = 'VISpl'
    # plot_vals, names_to_plot = get_subkeys_to_plot(df_means, aging_df, reg_name='Isocortex', dname='density_W', to_remove=to_remove, lvl_low=5, lvl_high=9)
    
    # all_avg = []
    # for i_n, reg_name in enumerate(names_to_plot):
    #     match = plot_vals[plot_vals['acronym'].str.fullmatch(reg_name) == True]
    #     baseline = match.iloc[np.where(match['age'] == 60)[0]]['density_W'].mean()
    #     match['fold_change'] = match['density_W']/baseline
        
    #     ### Also relate beginning density to fold change?
    #     avg_stats = match.groupby(['exp', 'acronym']).mean(numeric_only=True).reset_index()
    #     # copy the fold change over from 23mos to P60 so can plot later
    #     avg_stats.loc[np.where(avg_stats['exp'] == 'P60')[0], 'fold_change'] = avg_stats.loc[np.where(avg_stats['exp'] == 'P620')[0], 'fold_change'].values[0]
         
    #     avg_stats = avg_stats.loc[np.where(avg_stats['exp'] == 'P60')[0]]
        
    #     all_avg.append(avg_stats)
    
    
    # all_avg = pd.concat(all_avg)
    
    
    # plt.figure(figsize=(3.5,3))
    # fig = sns.regplot(data=all_avg, x='density_W', y='fold_change', color='gray')
    
    # #calculate slope and intercept of regression equation
    # slope, intercept, r, p, sterr = stats.linregress(x=fig.get_lines()[0].get_xdata(),
    #                                                        y=fig.get_lines()[0].get_ydata())
    
    # #display slope and intercept of regression equation
    # print(slope)

    # r,p = stats.pearsonr(all_avg.dropna()['density_W'], all_avg.dropna()['fold_change'])
    # print(r)
    # print(p)
    
    
    # all_texts = []
    # for line in range(0,all_avg.shape[0]):
    #      all_texts.append(plt.text(all_avg['density_W'].iloc[line]+0.01, all_avg['fold_change'].iloc[line], 
    #      all_avg['acronym'].iloc[line],
    #      ha='center', va='center',
    #      size=10, color='black', weight='normal'))
    # #plt.text(all_texts)
    
    # ax = plt.gca()
    # ax.legend().set_visible(False)
    # plt.yticks(fontsize=fontsize - 2)
    # plt.xticks(fontsize=fontsize - 2)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # #ax.spines['left'].set_visible(False)
    # #plt.legend(loc = 'upper left', frameon=False)
    # plt.ylim([1, 3])
    # plt.xlabel('Density at P60 (cells/mm\u00b3)', fontsize=fontsize)
    # plt.ylabel('Fold change at P650', fontsize=fontsize)
    
    # ### set scientific notation
    # ax.ticklabel_format(axis='x', scilimits=(-4, 4))
    # ax.xaxis.get_offset_text().set_fontsize(fontsize-2)
    # plt.xticks(np.arange(0, 15000, step=5000))
    
    
    
    # plt.tight_layout()
    
    # adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    # plt.savefig(sav_fold + exp_name + '_density_P60_vs_fold_change.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name + '_density_P60_vs_fold_change.svg', format='svg', dpi=300)
    
    
    # ### Declare plotting variables
    # palette = sns.color_palette("Set2")
    # ### CAREFUL ---> df_means right now is pooled from ALL groups... for sorting...
    
    
 
    # #%%% Compare layers for each cortical region
    # # names_to_plot = ['SSp', 'MOp','RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'ECT', 'PERI', 'ORB', 'TEa']
    # # names_to_plot = ['ACA','FRP', 'GU', 'ILA', 'PL', 'AI',  'VISC']  # 'PTLp' doesn't exist?
    # # names_to_plot = ['SSp', 'MOp','RSP', 'SSs', 'MOs', 'VIS', 'AUD', 'PERI', 'TEa', 'PL', 'ACA', 'AI']
    # names_to_plot = ['SSp', 'MOp', 'SSs', 'MOs', 'VIS', 'AUD', 'PERI', 'TEa', 'PL', 'ACA', 'AI'] #no RSP

    # palette = sns.color_palette("Set2", len(names_to_plot))
    # #styles = ['-', '--', '-.', ':']
    # styles=['-']
    
    # layers = ['1', '2/3', '4', '5', '6']
    
    # fontsize = 14
    
    # # all_layers = []

    # fig1, ax1 = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)
    # fig2, ax2 = plt.subplots(4, 3, figsize=(8, 8), sharex=True, sharey=True)
    # for i_x, name in enumerate(names_to_plot):

    #     for i_n, layer in enumerate(layers):
    #         # if name matches the first half at minimum, then go and plot
    #         match = aging_anya[aging_anya['acronym'].str.contains(name) == True]
    #         if name == 'RSP':  ### drop this weird dorsal layer 4 which is empty
    #             match = match[match['acronym'].str.contains('RSPd4') == False]   
    #         match=match.reset_index()
            

    #         ### DROP layer 1 from M127 and M126 for the moment due to delipidation artifiact
    #         # if layer == '1':
    #             #match = match.drop(index=np.where(match['dataset'] == 'M267')[0]).reset_index()
    #         #     #match = match.drop(columns=['level_0'])
    #         #     match = match.drop(index=np.where(match['dataset'] == 'M127')[0]).reset_index()
    #         #     match = match.drop(columns=['level_0'])
    #         #     match = match.drop(index=np.where(match['dataset'] == 'M126')[0]).reset_index()
    #         #     #match = match.drop(index=np.where(match['dataset'] == 'M310')[0])     

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                    
    #         lay_df = match[match['names'].str.contains(layer) == True]
            
    #         if len(lay_df) == 0:
    #             continue

    #         zero_df = {'exp':'LncOL1', 'age':0, 'density_W':0, 'density_NORM':0, 'num_OLs_W':0, 'atlas_vol_W':1}
    #         zero_df = pd.DataFrame(zero_df, index=[5])
            
    #         lay_df = pd.concat([lay_df, zero_df]).fillna(0)
            
            
    
    #         cur_ax = sns.lineplot(ax=ax1.flatten()[i_x], x=lay_df['age'], y=lay_df['density_W'], label=layer, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
    #                      errorbar='se',
    #                      marker='o',
    #                      markersize=8, linewidth=2).set(title=name) 
            
    #         lay_df['density_NORM'] = lay_df['density_W']/np.nanmean(lay_df.iloc[np.where(lay_df['exp'] == 'P60')[0]]['density_W'])


    #         sns.lineplot(ax=ax2.flatten()[i_x], x=lay_df['age'], y=lay_df['density_NORM'], label=layer, color=palette[i_n % len(palette)], linestyle=styles[i_n % len(styles)],
    #                      errorbar='se',
    #                      marker='o',
    #                      markersize=8, linewidth=2).set(title=name) 
    #         ax2.flatten()[i_x].set_ylim(0, 6)
            
            
            
    # ### Remove legends
    # for ax in ax1.flatten():
    #     ax.legend([],[], frameon=False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
        
    #     ax.set_xlabel('')
    #     ax.set_ylabel('')
        
    #     ax.set_ylim([0, 60000])
    #     ax.ticklabel_format(axis='y', scilimits=(-4, 4))
    #     ax.xaxis.get_offset_text().set_fontsize(fontsize-2)
        
    #     ax.tick_params(axis='x', labelsize=10)
    #     ax.tick_params(axis='y', labelsize=10)
        
        
    # for ax in ax2.flatten():
    #     ax.legend([],[], frameon=False)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)

    #     ax.set_xlabel('')
    #     ax.set_ylabel('')

        
    #     ax.tick_params(axis='x', labelsize=10)
    #     ax.tick_params(axis='y', labelsize=10)
        
        
        
    # handles, labels = ax1[0, 0].get_legend_handles_labels()
    # fig1.legend(handles, labels, loc='upper right')
    # fig2.legend(handles, labels, loc='upper right')


    
    # fig1.text(0.04, 0.5, 'Density (cells/mm\u00b3)', va='center', rotation='vertical', fontsize=fontsize)
    # fig2.text(0.04, 0.5, 'Normalized cell density', va='center', rotation='vertical', fontsize=fontsize)

    # plt.figure(fig1)
    # plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING.svg', format='svg', dpi=300)
    

    # plt.figure(fig2)
    # plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING_NORM.png', format='png', dpi=300)
    # plt.savefig(sav_fold + exp_name +'_by_layer_and_region_AGING_NORM.svg', format='svg', dpi=300)
    
    
    
    
    #%%% PLOT global regions of interest- Box and whisker plot- Used for manuscript!
    

    # names_to_plot = [
    #                   'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
    #                   'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
    #                   'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
    #                   'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

    #                   ]
  # ## Ordered by highest P30 value  

  #   names_to_plot = ['MO','SS', 'VIS','ACA','ORB','RSP','ECT','AUD',
  #                                       'fi', 'cc','hc',      ### Dorsal
  #                                      'int', 'amc','cpd','fx','sm', ###Ventral
  #                                      'In','IIn',
  #                                      'SSp'
                                      
  #                     ]
    
  #   palette = sns.color_palette("Set2", len(names_to_plot))

    
    
  #   plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX_wm',
  #                  ylim=3000, figsize=(7, 3))
    
    
    #%%% PLOT global regions of interest- For revisions
    # names_to_plot = [
    #                   'fi', 'cc','hc',      ### Dorsal
    #                  'int', 'amc','cpd','fx','sm', ###Ventral
    #                  'In','IIn',
    #                  'SSp'
                     
    #                  ]

    #%% All box and whister plots for manuscript

    #%%%% Broad regions- Currently panel E
    names_to_plot = ['HY', 'TH','MB','HPF','Isocortex']
                     
                     
    
    palette = sns.color_palette("Set2", len(names_to_plot))
    
    
    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_all_grey_reg',
                   ylim=2000, figsize=(3, 2.5))


#%%%%% Broad regions statistics
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
 
    
    names_to_plot = ['HY', 'TH','MB','HPF','Isocortex']
    temp_stats = aging_anya
    all_melted = []    
    for i_n, name in enumerate(names_to_plot):
        # if name matches FULLY, then go and plot
        match = aging_anya[aging_anya['acronym'].str.fullmatch(name) == True]

        all_melted.append(match)
        
       
    df_melt_stats = pd.concat(all_melted)
    df_melt_stats = df_melt_stats[df_melt_stats['density_W'].notna()]
    
    model = ols('density_W ~ C(age) + C(acronym) + C(age):C(acronym)', data = df_melt_stats).fit()
    
    anova_result = sm.stats.anova_lm(model, type = 2)
    
    results_list = []
    
    for acr in df_melt_stats['acronym'].unique():
        sub = df_melt_stats[df_melt_stats['acronym'] == acr]
        if sub['age'].nunique() <= 1:
            continue
        tukey = pairwise_tukeyhsd(sub['density_W'], sub['age'])
        #Converting it to a dataframe
        tbl = pd.DataFrame(data=tukey._results_table.data[1:], columns = tukey._results_table.data[0])
        tbl['acronym'] = acr
        for c in ['meandiff','p-adj','lower','upper']:
                tbl[c] = pd.to_numeric(tbl[c])
                tbl['reject']= tbl['reject'].map({'True':True,'False':False})
                results_list.append(tbl)
                
    if results_list:
        df_results = pd.concat(results_list,ignore_index=True)
        df_results = df_results.drop_duplicates(subset=['acronym','group1','group2','meandiff','p-adj'])
        df_results.to_csv(sav_fold + 'tukey_age_within_region_broad_gm_regions.csv', index=False)

    # res = pd.concat([model.params,model.pvalues],axis=1)
    # res.columns=['coefficient','pvalues']
    # #res = res[res.index.str.contains('age')]
    # res['corrected_p'] = multipletests(res['pvalues'],method="sidak")[1]
    
    #%%%% grey_white- Currently panel d
    names_to_plot = ['grey', 'fiber tracts']
                     
                     
    palette = sns.color_palette("Set2", 4)
    
    
    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_grey_white',
                   ylim=1500, figsize=(3, 3))
    #%%%%% grey_white_statistics- Currently panel d

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
 
    
    names_to_plot = ['grey', 'fiber tracts']
    temp_stats = aging_anya
    all_melted = []    
    for i_n, name in enumerate(names_to_plot):
        # if name matches FULLY, then go and plot
        match = aging_anya[aging_anya['acronym'].str.fullmatch(name) == True]

        all_melted.append(match)
        
       
    df_melt_stats = pd.concat(all_melted)
    df_melt_stats = df_melt_stats[df_melt_stats['density_W'].notna()]
    
    model = ols('density_W ~ C(age) + C(acronym) + C(age):C(acronym)', data = df_melt_stats).fit()
    
    anova_result = sm.stats.anova_lm(model, type = 2)
    print(anova_result)
    df_results.to_csv(sav_fold + 'tukey_age_within_region_broad_gm_wm_regions.csv', index=False)
    
    results_list = []
    
    for acr in df_melt_stats['acronym'].unique():
        sub = df_melt_stats[df_melt_stats['acronym'] == acr]
        if sub['age'].nunique() <= 1:
            continue
        tukey = pairwise_tukeyhsd(sub['density_W'], sub['age'])
        #Converting it to a dataframe
        tbl = pd.DataFrame(data=tukey._results_table.data[1:], columns = tukey._results_table.data[0])
        tbl['acronym'] = acr
        for c in ['meandiff','p-adj','lower','upper']:
                tbl[c] = pd.to_numeric(tbl[c])
                tbl['reject']= tbl['reject'].map({'True':True,'False':False})
                results_list.append(tbl)
                
    if results_list:
        df_results = pd.concat(results_list,ignore_index=True)
        df_results = df_results.drop_duplicates(subset=['acronym','group1','group2','meandiff','p-adj'])
        df_results.to_csv(sav_fold + 'tukey_age_within_region_broad_gm_wm_regions.csv', index=False)

    #%%%% All cortical regions- Currently panel g
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', #'AUDd', 'AUDpo', 'AUDv', 
                     'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', #'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(aging_anya, names_to_plot, palette,  sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX',
                   dname='density_W', dropna=True,
                   ylim=1250, figsize=(7, 2.6))
    

    #%%%% Major White Matter areas and two representative subdivisions- LFBS and CM
    names_to_plot = [
                     'scwm','fxs','lfbs','mfbs','eps','cm','mfbs', #Broad regions
                     'cc','lfbst', #Regions of the lfbs
                     'In','IIn','IIIn' #Cranial nerves
                     
                     
                     ]
    
    palette = sns.color_palette("Set2", len(names_to_plot))
    
    
    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_all_reg',
                   ylim=1750, figsize=(5.5, 3))

    
    #Cranial nerves and corpus callosum alone- Not used
    names_to_plot = [
                      'cc','lfbst','In','IIn','IIIn'
                     
                     ]
    
    palette = sns.color_palette("Set2", len(names_to_plot))
    
    
    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_lfbs_cranial_nerves',
                   ylim=1750, figsize=(4, 3))
    
    
    # #Cranial nerves
    # names_to_plot = [
    #                   'sm'
                     
    #                  ]
    
    # palette = sns.color_palette("Set2", len(names_to_plot))
    
    
    # plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_sm',
    #                ylim=3000, figsize=(7, 3))
    
    #looking at the CC subdivisions
    names_to_plot = [
                      'cc','fa','ccg','fp','ccb','ccs'
                     
                     ]
    
    palette = sns.color_palette("Set2", len(names_to_plot))
    

    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_cc_divisions',
                   ylim=3000, figsize=(7, 3))
    
    #%%%% looking at the HPC subdivisions
    names_to_plot = [
                      'alv','df','fi','fxpo','hc'
                     
                     ]
    
    palette = sns.color_palette("Set2", 4)
    

    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_hpc_divisions',
                   ylim=3000, figsize=(4.5, 3))
    
    
    #looking at the FXPO subdivisions
    names_to_plot = [
                      'mct','fx'
                     
                     ]
    
    palette = sns.color_palette("Set2", len(names_to_plot))
    

    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_fxpo_divisions',
                   ylim=3000, figsize=(7, 3))


    #broader regions
    names_to_plot = ['Isocortex', 
                    
                     'fiber tracts', 
                 
                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))
    

    plot_global(aging_anya, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_white_matter_reg_cortex_fibertracks_divisions',
                   ylim=3000, figsize=(2, 3))
    
    
    # plot raw numbers
    names_to_plot = [
                      'ACA', 'ORB', 'FRP', 'GU',  'ILA', 'PL', 'AI',      ### Frontal lobe
                     'RSP', 'SSp', 'SSs', 'MOp', 'MOs', 'PTLp',          ### Parietal lobe
                     'AUDp', 'AUDd', 'AUDpo', 'AUDv', 'VISC', 'TEa', 'ECT', 'PERI',               ### Temporal lobe
                     'VISp', 'VISal', 'VISam', 'VISl', 'VISli', 'VISpl', 'VISpm', 'VISpor', 'VISrl'                           # occipital lobe

                     ]
    palette = sns.color_palette("Set2", len(names_to_plot))

        
    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CORTEX_RAW_COUNT', dname='num_OLs_W',
                   ylim=600000, 
                   figsize=(7, 3),
                   logscale=True)
    
    
    # plt.ylim([0, 100000])
    # plt.tight_layout()
    
    
    names_to_plot = [
                     'CA1', 'CA2', 'CA3', 'DG-mo', 'DG-po', 'DG-sg',   ### Hippocampus

                        ### Retrohippocampal regions
                      #'RHP', 
                      'ENT', 'ENTl', 'ENTm', #'ENTmv',    ### Entorhinal areas divided into layer 1 - 6!!!
                      # 'ENTl1', 'ENTl2', 'ENTl3', 'ENTl5', 'ENTl6a',
                      # 'ENTm1', 'ENTm2', 'ENTm3', 'ENTm5', 'ENTm6',
                      
                        'PAR', #'PRE',  
                      
                      ### Para- post- and pre-subiculum (also divided into layer 1, 2, 3!!!)
                      
                      
                      ### Areas below here are too high to fit on ylim... need 40000
                       # 'POST',
                       # 'SUB', #'SUBv', 'SUBd'   ### subiculum

                     ]

    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_HIPPO',
                   ylim=25000, figsize=(4.2, 2.5))
    
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    
    ### stats for comparison
    # Compare Entorhinal areas
    match = aging_df[aging_df['acronym'].str.fullmatch('ENTl') == True]
    ENTL60 = match[match['age'] == 60]
    ENTL620 = match[match['age'] == 620]
        
    match = aging_df[aging_df['acronym'].str.fullmatch('ENTm') == True]
    ENTm60 = match[match['age'] == 60]    
    ENTm620 = match[match['age'] == 620]    
    
    
    print(stats.sem(ENTL60['density_W']))   
    print(stats.sem(ENTL620['density_W']))   
    print(stats.sem(ENTm60['density_W']))   
    print(stats.sem(ENTm620['density_W']))   
    

    tstat, p = stats.ttest_ind(ENTL60['density_W'], ENTm60['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('ENTL60:' + str(np.mean(ENTL60['density_W'])) + ' ENTm60: ' + str(np.mean(ENTm60['density_W'])))

    tstat, p = stats.ttest_ind(ENTL620['density_W'], ENTm620['density_W'], equal_var=True, alternative='two-sided')
    print(p)
    print('ENTL620:' + str(np.mean(ENTL620['density_W'])) + ' ENTm620: ' + str(np.mean(ENTm620['density_W'])))
  
    
    
        
        
    
        
    names_to_plot = [
                     'CBX', 'VERM', #vermal regions ### Cerebellum

                        ### Hemispheric regions, 
                      'HEM', 'SIM', 'AN', 'PRM', 'COPY', 'PFL', 'FL',    
                      
                      # 'FN', 'IP', 'DN',   ### Cerebellar nuclei
                      # 'arb',   ### arbor vitae

                     ]

    plot_global(aging_df, names_to_plot, palette, sav_fold, exp_name + '_GLOBAL_COMPARISON_CEREBELLUM',
                   ylim=25000, figsize=(4.2, 2.5))
    
    
#%% Writing csv and pkl for Jacob- Correlation analyses    
aging_anya.to_csv(sav_fold+'aging_dataset_for_jacob.csv')

import pickle as pkl
with open(sav_fold+'aging_dataset_for_jacob.pkl','wb') as file:
    pkl.dump(aging_anya, file)
        
    