# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:19:05 2026

@author: User
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats



#Import CSV from all cortical samples
sav_dir= "X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched" 
sample2_3 = pd.read_excel(sav_dir + "/L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_(ii2_L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_Image_3)_complete.xlsx")

sample2_8 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_(ii7_L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_Image_8)_complete.xlsx")
sample2_8["Mouse_ID"] = "2"
sample2_8["Sample_ID"] = "8"

sample2_11 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_(ii10_L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_Image_11)_complete.xlsx")
sample2_11["Mouse_ID"] = "2"
sample2_11["Sample_ID"] = "11"

sample2_14 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_(ii13_L004_2f_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-02_Image_14)_complete.xlsx")
sample2_14["Mouse_ID"] = "2"
sample2_14["Sample_ID"] = "14"

sample4_1_3 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_4_1_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-09_(ii2_L004_4_1_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-09_Image_3)_complete.xls")
sample4_1_3["Mouse_ID"] = "4"
sample4_1_3["Sample_ID"] = "1_3"


sample4_2_3 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_4_2_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-07_(ii2_L004_4_2_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-07_Image_3)_complete.xls")
sample4_2_3["Mouse_ID"] = "4"
sample4_2_3["Sample_ID"] = "2_3"

sample5_b_3 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_5m_bottom_right_corner_inverted_405_dapi_488_egfp_546_lnc...rted_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-08.czi #3_complete.xls")
sample5_b_3["Mouse_ID"] = "5"
sample5_b_3["Sample_ID"] = "b_3"

sample5_o_3 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_5m_only_use_first_4_scenes_top_left_corner_inverted_405_d...rted_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-10.czi #3_complete.xls")
sample5_o_3["Mouse_ID"] = "5"
sample5_o_3["Sample_ID"] = "o_3"

sample11_b_3 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_11f_bpttp,_l_corner_inverted_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-04_Image_3_complete.xls")
sample11_b_3["Mouse_ID"] = "11"
sample11_b_3["Sample_ID"] = "b_3"

sample11_t_3 = pd.read_excel("X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/L004_11f_top_l_corner_inverted_405_dapi_488_egfp_546_lncol1_657_egfp-Stitching-03_Image_3_complete.xls")
sample11_t_3["Mouse_ID"] = "11"
sample11_t_3["Sample_ID"] = "t_3"

def pre_process_csv (df):
    df = df.drop([0])
    df["mean_intensity_3"] = df["Intensity Mean Ch=3 Img=1"].astype(float)
    df["mean_intensity_4"] = df["Intensity Mean Ch=4 Img=1"].astype(float)
    return df

sample2_8 = pre_process_csv(sample2_8)
sample2_11 = pre_process_csv(sample2_11)
sample2_14 = pre_process_csv(sample2_14)
sample4_1_3 = pre_process_csv(sample4_1_3)
sample4_2_3 = pre_process_csv(sample4_2_3)
sample5_b_3 = pre_process_csv(sample5_b_3)
sample5_o_3 = pre_process_csv(sample5_o_3)
sample11_b_3 = pre_process_csv(sample11_b_3)
sample11_t_3 = pre_process_csv(sample11_t_3)


collected = [sample2_8,
             sample2_11,
             sample2_14,
             sample4_1_3,
             sample4_2_3,
             sample5_b_3,
             sample5_o_3,
             sample11_b_3,
             sample11_t_3]
merged = pd.concat(collected)

z_3 = np.abs(stats.zscore(merged['mean_intensity_3']))
z_4 = np.abs(stats.zscore(merged['mean_intensity_4']))
mask = (z_3 <= 10) & (z_4 <= 10)
merged_z = merged.loc[mask].copy()
#The above code is just removing the two outlier points of data given their large z score
#As a note, it is only removin two out of 3,567 cells


sns.regplot(data= merged, x="mean_intensity_3", y= "mean_intensity_4", scatter_kws=dict(s=0.2,alpha=0.5,color = "forestgreen"), line_kws= {"color":"k", "linewidth" : 1}, truncate = False )
sns.despine()
ax = plt.gca()
ax.xaxis.set_major_locator(mticker.MultipleLocator(1000))
ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
ax.set(xlabel=None)
ax.set(ylabel=None)
ax.set_xlim(0,6000)
ax.set_ylim(0,4000)
plt.xticks(rotation=45, ha="right") 
fig = plt.gcf()
fig.set_size_inches(4,2.8, forward=True)
plt.tight_layout()
#plt.savefig(r'X:/Bergles lab/Anya/800/20250428_HCR_Rik_spec_P30/only_stitched/egfp_lnc_scatter.svg', format="svg", bbox_inches="tight")
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_z['mean_intensity_3'],merged_z['mean_intensity_4'])
r_sqr = r_value**2
print(" r_sqr =", r_sqr)
print(" r =", r_value)
print(" p_value =", p_value)
print(" slope =", slope)
print(" intercept =", intercept)
print("test with outliers")
slope, intercept, r_value, p_value, std_err = stats.linregress(merged['mean_intensity_3'],merged['mean_intensity_4'])
r_sqr = r_value**2
print(" r_sqr =", r_sqr)
print(" p_value =", p_value)
#merged_z.to_csv(sav_dir + "/merged_csv.csv")

##Results copied from 2/18/2026:
   #  r_sqr = 0.6910307512646734
   #  r = 0.8312825941066452
   #  p_value = 0.0
   #  slope = 0.6126863004695137
   #  intercept = 41.88586771245582
   #3567 cells in 
   # test with outliers
   #  r_sqr = 0.6372417085511081
   #  p_value = 0.0