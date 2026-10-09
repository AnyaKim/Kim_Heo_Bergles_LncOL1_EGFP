# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 15:57:23 2025

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'    

fontsize = 11

#Importing each individual file

sav_dir= r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration' #All_folders

#Importing data and adding additional column IDs
##These are extended tracks from the original nTnG analysis, but with added tracking on either side of each EGFP event
##Nuclei were tracked when they were EGFP- to allow us to analyze the complete track
##also saved in utf8_format

f4_fov1_0514 = pd.read_csv(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\20240514_F4_FOV1\20240514_F4_FOV1_allspots_extended_tracks_utf8.csv')
f4_fov1_0514 = f4_fov1_0514.drop([0,1,2])
f4_fov1_0514["Date"] = "20240514"
f4_fov1_0514["Mouse_ID"] = "F4"
f4_fov1_0514["FOV"] = "FOV1"

f4_fov4_0514 = pd.read_csv(r"C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\20240514_f4_FOV4\20240514_F4_FOV4_allspots_extended_tracks_utf8.csv")
f4_fov4_0514 = f4_fov4_0514.drop([0,1,2])
f4_fov4_0514["Date"] = "20240514"
f4_fov4_0514["Mouse_ID"] = "F4"
f4_fov4_0514["FOV"] = "FOV4"


f1_fov1_0719 = pd.read_csv(r"C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\20240719_F1_FOV1\20240719_F1_FOV1_allspots_extended_tracks_with_singlets_utf8.csv")
f1_fov1_0719 = f1_fov1_0719.drop([0,1,2])
f1_fov1_0719["Date"] = "20240719"
f1_fov1_0719["Mouse_ID"] = "F1"
f1_fov1_0719["FOV"] = "FOV1"


m1_fov3_0810 = pd.read_csv(r"C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\20240810_M1_FOV3\20240810_M1_FOV3_after_fiji_REGISTERED_FULLSTACK-bdv-mamut_spots_extended_with_singlets_utf8.csv")
m1_fov3_0810 = m1_fov3_0810.drop([0,1,2])
m1_fov3_0810["Date"] = "20240810"
m1_fov3_0810["Mouse_ID"] = "M1"
m1_fov3_0810["FOV"] = "FOV3"

m4_fov3_0810 = pd.read_csv(r"C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\20240810_M4_FOV3\20240810_M4_FOV3_after_fiji_REGISTERED_FULLSTACK-bdv-mamut_updated_allspots1_extended_tracks_utf8.csv")
m4_fov3_0810 = m4_fov3_0810.drop([0,1,2])
m4_fov3_0810["Date"] = "20240810"
m4_fov3_0810["Mouse_ID"] = "M4"
m4_fov3_0810["FOV"] = "FOV3"

def pre_process_csv (df):
    df["MEAN_INTENSITY_CH1"] = df["MEAN_INTENSITY_CH1"].astype(float)
    df["POSITION_T"] = df["POSITION_T"].astype(float)
    df["TRACK_ID"] = df["TRACK_ID"] + df["Mouse_ID"] + df["FOV"]
    filtered_df = df.groupby("TRACK_ID")
    
    #Include only tracks which end with d or OL
    filtered_df = filtered_df.filter(lambda x: x['LABEL'].isin(['d','OL']).any())
    
    #Also label all tracks by if they're dead or not dead
    filtered_df['fate'] = (
        filtered_df.groupby('TRACK_ID')['LABEL']
        .transform('last')              # last label in each track
        .map({'d': 'dead', 'OL': 'ol'}) # map to fate
        )

    #Find the timpoint at which the mean EGFP intensity is highest
    idx = filtered_df.dropna(subset=['MEAN_INTENSITY_CH1']).groupby('TRACK_ID')['MEAN_INTENSITY_CH1'].idxmax()
    time_at_max = filtered_df.loc[idx].set_index('TRACK_ID')['POSITION_T']
    filtered_df['time_max'] = filtered_df['TRACK_ID'].map(time_at_max)
    
    #Get relative time, where max EGFP intensity is 0, negative is before and positive is after
    filtered_df['relative_time'] = filtered_df['POSITION_T'] - filtered_df['time_max']
    
    #Remove all tracks in which there are less than 5 days before the event itself
    filtered_df = filtered_df.groupby('TRACK_ID').filter(lambda x: x['relative_time'].min() <-5)
    
    #Select only values within 5 days of the EGFP peak
    filtered_df = filtered_df.loc[filtered_df['relative_time'].between(-5, 5, inclusive='both')]

    #Get relative EGFP intensity- divide by 25th percentile of the value of the track 
    filtered_df["baseline_intensity"] = filtered_df.groupby("TRACK_ID")["MEAN_INTENSITY_CH1"].transform(lambda x: x.quantile(.25))
    
    #Divide by minimum to normalize
    filtered_df['normalized_intensity'] = (filtered_df['MEAN_INTENSITY_CH1'] - filtered_df['baseline_intensity'])/filtered_df['baseline_intensity']
    
    return filtered_df
        
#Processing individual files
f1_fov1_0719_done = pre_process_csv(f1_fov1_0719)
f4_fov1_0514_done = pre_process_csv(f4_fov1_0514)
f4_fov4_0514_done = pre_process_csv(f4_fov4_0514)
m1_fov3_0810_done = pre_process_csv(m1_fov3_0810) 
m4_fov3_0810_done = pre_process_csv(m4_fov3_0810)

#Merge all files
collected = [f1_fov1_0719_done,f4_fov1_0514_done,f4_fov4_0514_done,m1_fov3_0810_done,m4_fov3_0810_done]
merged = pd.concat(collected)

merged.to_csv(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\merged_all_tracks_python_output.csv')


#Plot the EGFP intensity values across time
sns.lineplot(data=merged, x="relative_time", y="normalized_intensity", hue = "fate", palette = ["#40923A","#B3D495"])
plt.xlim(-5,5)
plt.ylim(-.5,2)
ax = plt.gca()
ax.set_xlabel("Relative Time")
ax.set_ylabel("Normalized Intensity")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
 
plt.yticks(fontsize=fontsize - 2)
plt.xticks(fontsize=fontsize - 2)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=11))
 
 
plt.tight_layout()
 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])

plt.legend(frameon=False)

fig1 = plt.gcf()
fig1.set_size_inches(2.4,2 , forward=True)
plt.tight_layout()
plt.savefig(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\intensity_over_time.svg', format="svg", bbox_inches="tight")
plt.show()

#Plot the EGFP intensity values across time- Individual traces- dead only
sns.lineplot(data=merged[merged['fate'] == 'dead'], x="relative_time", y="normalized_intensity", units = "TRACK_ID", hue = 'fate', palette = ["#B3D495"], estimator = None, alpha = 0.25, linewidth = 0.5)
plt.xlim(-5,5)
plt.ylim(-2,6)
ax = plt.gca()
ax.set_xlabel("Relative Time")
ax.set_ylabel("Normalized Intensity")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
 
plt.yticks(fontsize=fontsize - 2)
plt.xticks(fontsize=fontsize - 2)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=11))
 
 
plt.tight_layout()
 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])

plt.legend(frameon=False)

fig1 = plt.gcf()
fig1.set_size_inches(2.4,2 , forward=True)
plt.tight_layout()
plt.savefig(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\intensity_over_time_traces_dead.svg', format="svg", bbox_inches="tight")
plt.show()


#Plot the EGFP intensity values across time- Individual traces
sns.lineplot(data=merged[merged['fate'] == 'ol'], x="relative_time", y="normalized_intensity", units = "TRACK_ID", hue = 'fate', palette = ["#40923A"], estimator = None, alpha = 0.25, linewidth = 0.5)
plt.xlim(-5,5)
plt.ylim(-2,6)
ax = plt.gca()
ax.set_xlabel("Relative Time")
ax.set_ylabel("Normalized Intensity")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
 
plt.yticks(fontsize=fontsize - 2)
plt.xticks(fontsize=fontsize - 2)
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=11))
 
 
plt.tight_layout()
 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])

plt.legend(frameon=False)

fig1 = plt.gcf()
fig1.set_size_inches(2.4,2 , forward=True)
plt.tight_layout()
plt.savefig(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\intensity_over_time_traces_alive.svg', format="svg", bbox_inches="tight")
plt.show()

##Plot total number of cells across time
fill_to = 5        # extend zeros up to this time
start_time = -5    # set explicitly, or use merged['relative_time'].min()

##Create a grid of time points
mouse_ids = merged['Mouse_ID'].unique()
time_points = np.arange(start_time, fill_to + 1)
full_index = pd.MultiIndex.from_product([mouse_ids, time_points],
names=['Mouse_ID', 'relative_time'])

##Dead counts per mouse/time, reindexed to include missing times with zeros
dead_counts = (
merged[merged['fate'] == 'dead']
.groupby(['Mouse_ID', 'relative_time'])
.size()
.reindex(full_index, fill_value=0)
.rename('dead_n')
)

#Calculate the baseline number of cells at timepoint -5
merged_dead = merged[merged['fate'] == 'dead']
baseline = (
merged_dead[merged_dead['relative_time'] == -5]
.groupby('Mouse_ID')
.size()
.rename('baseline_n')
)

#Computing actual ratio of living cells
out = (
dead_counts.to_frame()
.assign(baseline_n=lambda df: df.index.get_level_values('Mouse_ID').map(baseline))
.assign(pct_dead=lambda df:( (df['dead_n'] / df['baseline_n'])*-100)+100)
.reset_index()
)

out.to_csv(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\dead_tracks_processed.csv', index=False)

#Plot percentage of dead cells over time
sns.barplot(
    data=out,   
    x='relative_time',
    y='pct_dead',
    color = "#B3D495",
    errorbar='se',
    errwidth=1,
    capsize = .1
    )

sns.stripplot(
    data=out, 
    x='relative_time',
    y='pct_dead',
    color = "#000000",
    size = 3
    )

ax = plt.gca()
ax.set_xlabel("Relative Time")
ax.set_ylabel("Percent Dead")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
 
plt.yticks(fontsize=fontsize - 2)
plt.xticks(fontsize=fontsize - 2)
 
 
plt.tight_layout()
 
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])

fig = plt.gcf()
fig.set_size_inches(2.4,2 , forward=True)
plt.tight_layout()
plt.savefig(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\percent_dead_over_time.svg', format="svg", bbox_inches="tight")

#One more plot for Max EGFP intensity successful vs dead
##Calculating max intensity per track
merged_max_int = merged.groupby(['Mouse_ID','FOV','TRACK_ID','fate']).agg(
    max_int = ("normalized_intensity",'max')
    )

##Averaging that by mouse by fate
merged_max_int = merged_max_int.groupby(['Mouse_ID','fate']).agg(
    max_int_average = ("max_int",'mean')
    )

##Export for Prism
 
merged_max_int.to_csv(r'C:\Users\User\OneDrive - Johns Hopkins\Bergles\Imaging Data\710_registration\Complete_registration\for_max_int_calc.csv')


 