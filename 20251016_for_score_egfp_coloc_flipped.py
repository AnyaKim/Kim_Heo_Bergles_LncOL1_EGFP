# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:53:53 2025

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Opening Files

# Update save directory with specific project 
sav_dir= r"D:\710_in_vivo_temp_processing\710\Pup_surgery_score_lncol1\round1_mnt_fov5\processed" #R1_MNT_FOV5
df_score_endpoints = pd.read_csv(sav_dir + r"\round1_mnt_fov5_score_spots.csv")
df_EGFP_events = pd.read_csv(sav_dir + r"\round1_mnt_fov5_lnc_spots.csv")
xy_division = 2.47003029904  # R1_MNT_FOV3 1500 pixels for 607.28 um

#sav_dir= r"D:\710_in_vivo_temp_processing\710\Pup_surgery_score_lncol1\round2_mr2_fov3\for_reg" #R2_mr2_fov3
#df_score_endpoints = pd.read_csv(sav_dir + r"\round2_mr2_fov5_score_spots.csv")
#df_EGFP_events = pd.read_csv(sav_dir + r"\round2_mr2_fov5_egfp_spots.csv")
#xy_division = 0.8431036754 # 607.28 um and 512 pixels in R2_MR2_FOV3

# Delete first couple of rows since they're just headers
df_EGFP_events = df_EGFP_events.drop([0,1,2])
df_score_endpoints = df_score_endpoints.drop([0,1,2])

# %% Micron conversion

#Double check that these are the correct values for your given FOV!
z_step_size = 2.0          # Z (steps -> microns) multiply by this

df_EGFP_events["um_X"] = df_EGFP_events["POSITION_X"].astype(float)/xy_division
df_EGFP_events["um_X"] = (df_EGFP_events["um_X"] - 607.28)*-1
df_EGFP_events["um_Y"] = df_EGFP_events["POSITION_Y"].astype(float)/xy_division
df_EGFP_events["um_Y"] = (df_EGFP_events["um_Y"] - 607.28)*-1
df_EGFP_events["um_Z"] = df_EGFP_events["POSITION_Z"].astype(float)*z_step_size


df_score_endpoints["um_X"] = df_score_endpoints["POSITION_X"].astype(float)/xy_division
df_score_endpoints["um_Y"] = df_score_endpoints["POSITION_Y"].astype(float)/xy_division
df_score_endpoints["um_Z"] = df_score_endpoints["POSITION_Z"].astype(float)*z_step_size

# %% Parsing data

# Keep only transient events (white spots)
df_EGFP_events = df_EGFP_events[df_EGFP_events['MANUAL_SPOT_COLOR'].isin(['r=255;g=255;b=255',"r=255;g=255;b=0"]) == True]

# Parse labels of SCoRe endpoints
endpoints_filter = df_score_endpoints['LABEL'].str.contains("ID")
df_score_endpoints = df_score_endpoints[~endpoints_filter]
df_score_endpoints[['sheath_id','endpoint_id']] = df_score_endpoints.LABEL.str.split(".", expand = True)

#keep the 1s and 2s separate
df_score_endpoints_1 = df_score_endpoints[df_score_endpoints['endpoint_id'].isin(['1']) == True]
df_score_endpoints_1 = df_score_endpoints_1.reset_index()
df_score_endpoints_1 = df_score_endpoints_1.sort_values("sheath_id")
df_score_endpoints_2 = df_score_endpoints[df_score_endpoints['endpoint_id'].isin(['2']) == True]
df_score_endpoints_2 = df_score_endpoints_2.reset_index()
df_score_endpoints_2 = df_score_endpoints_2.sort_values("sheath_id")

df_score_midpoints = pd.DataFrame()

#Find the midpoints of each sheath

df_score_midpoints["um_X"] = (df_score_endpoints_1["um_X"] + df_score_endpoints_2["um_X"])/2
df_score_midpoints["um_Y"] = (df_score_endpoints_1["um_Y"] + df_score_endpoints_2["um_Y"])/2
df_score_midpoints["um_Z"] = (df_score_endpoints_1["um_Z"] + df_score_endpoints_2["um_Z"])/2 

df_score_midpoints["sheath_id"] = (df_score_endpoints_1["sheath_id"])
df_score_midpoints["POSITION_T"] = (df_score_endpoints_1["POSITION_T"])
 
# %% Looking at the EGFP

def find_nearest_egfp(df_score_midpoints, df_EGFP_events, t_column="POSITION_T", lookback_days=1):
    xyz_columns=["um_X", "um_Y", "um_Z"]

    # Convert XYZ columns to numpy arrays for efficient computation
    egfp_xyz = df_EGFP_events[xyz_columns]#.to_numpy()
    egfp_t = df_EGFP_events[t_column]#.to_numpy()
    
    nearest_egfp = pd.DataFrame()
    
    match_rows=[]
    # Iterate over each row of the midpoint dataframe
    for _, mid_row in df_score_midpoints.iterrows():
        # mid_row is the current row being looked at
        
        # Get the time for this row
        t_mid = mid_row[t_column]
        # Get the "min" time we want to look at. We have to add because units are "T-" since time 0 #question
        t_min = max(pd.to_numeric(t_mid) - lookback_days,0)
        # Set the "max" time we want to look at as the current midpoint's time
        t_max = pd.to_numeric(t_mid)

        # Find EGFP events within time window. 
        # time_mask will be a Pandas series with values of True for the condition we care about 
        # (w/in 3 days of current time), else false.
        time_mask = (pd.to_numeric(egfp_t) >= t_min) & (pd.to_numeric(egfp_t) <= t_max)
        
        # Use the time mask to get egfp positions and times that are within the time period of interest
        candidates_xyz = egfp_xyz[time_mask]
        # candidates_t = egfp_t[time_mask]
        # candidates_idx = np.where(time_mask)[0]

        # if len(candidates_xyz) == 0:
            # nearest_indices.append(None)
            # nearest_distances.append(None)
            # nearest_time_differences.append(None)
            # continue

        # Calculate distances in XYZ space
        # First get the midpoint positions for the calculation
        midpoint_xyz = mid_row[xyz_columns].to_numpy()
        dists = np.sqrt(np.square((candidates_xyz - midpoint_xyz).astype(float)).sum(axis=1))
        
        # Find the index of closest egfp
        closest_index = dists.idxmin()
        closest_label =df_EGFP_events.loc[closest_index].LABEL
        closest_time = df_EGFP_events.loc[closest_index].POSITION_T
        closest_dist = dists.min()
        match_df = pd.DataFrame({'LABEL': [closest_label], 'dist': [closest_dist], 'egfp_POSITION_T': [closest_time], 'mid_T':[t_mid], 'sheath_id': [mid_row.sheath_id]})
        match_rows.append(match_df)
        
        # pd.concat([match_labels, dists, match_times], axis=1)
        
        # min_idx = np.argmin(dists)
        # nearest_indices.append(candidates_idx[min_idx])
        # nearest_distances.append(dists[min_idx])
        # nearest_time_differences.append(candidates_t[min_idx] - t_mid)
        # distance, label, time of egfp - for all w/in 3 days, and w/in 70um

    # Add results as new columns
    # df_score_midpoints = df_score_midpoints.copy()
    # df_score_midpoints["EGFP_nearest_index"] = nearest_indices
    # df_score_midpoints["EGFP_nearest_distance"] = nearest_distances
    # df_score_midpoints["EGFP_nearest_dT"] = nearest_time_differences
    nearest_egfp = pd.concat(match_rows)
    return nearest_egfp

# Example usage:
result = find_nearest_egfp(df_score_midpoints, df_EGFP_events)
print('donezo')

result.to_excel(sav_dir + r"\nearest_neighbor_flip.xlsx")

print (result["dist"].mean())

result["dist"].hist(bins=20)
plt.title("Histogram of nearest neighbor distance")
plt.xlabel('Distance to nearest neighbor (um)')
plt.ylabel('Number of sheaths')
