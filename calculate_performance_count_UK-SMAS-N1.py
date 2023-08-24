# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:16:31 2022

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement
from ecosound.evaluation.prf import PRF
import numpy as np

###############################################################################
##  input parameters ##########################################################
###############################################################################

# Annotation file:
annot_files=[]
annot_files.append(r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc")
annot_files.append(r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20210522\Annotations_dataset_UK-SAMS-WestScotland-202105-N1 annotations_withSNR.nc")

# Detection folder or detection sqllite file
detec_file = r"D:\NOAA\2022_Minke_whale_detector\results\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm\UK-SAMS-N1\detections.sqlite"

# output folder
out_dir = r"D:\NOAA\2022_Minke_whale_detector\results\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm\UK-SAMS-N1\performance_results"

# restrictions on min and max dates:
date_min = '2020-11-02 00:00:00'  # date string (e.g. '2020-11-02 00:00:00') or None
date_max = '2021-06-06 00:00:00'  # # date string (e.g. '2020-11-02 00:00:00') or None


# Detection threshold to test
thresholds = np.arange(0, 1, 0.1)
# thresholds = [0.9]

# Name of the class to test
target_class = "MW"

# List of files to use
files_to_use = "both"  # 'detec', 'annot', 'both', list


# Beta parameter for calculating the F-score
F_beta = 1

# Matching criteria
freq_ovp = True
dur_factor_max = None
dur_factor_min = None
ovlp_ratio_min = None
remove_duplicates = False
inherit_metadata = False
filter_deploymentID = False
do_plot = True

###############################################################################
###############################################################################

# load ground truth data
print(" ")
print("Loading manual annotations...")
for idx, annot_file in enumerate(annot_files):
    annot_tmp = Measurement()
    annot_tmp.from_netcdf(annot_file)
    if idx == 0:
        annot = annot_tmp
    else:
        annot = annot + annot_tmp
print(annot)

print("Annotation labels:")
print(annot.get_labels_class())

# remove annotation that are too uncertain
print('filtering annotation by analyst confidence')
annot.filter('confidence==1', inplace=True)
print(annot)

# load detections
print(" ")
print("Loading automatic detections...")
detec = Annotation()
detec.from_sqlite(detec_file)
print(detec)

# merge overlapped detections
detec.merge_overlapped(time_tolerance_sec=2, inplace=True)

# merge overlapped annotations

# calculate performance
print(" ")
print("Calculating performance...")
PRF.count(
    annot=annot,
    detec=detec,
    out_dir=out_dir,
    target_class=target_class,
    files_to_use=files_to_use,
    date_min=date_min,
    date_max=date_max,
    thresholds=thresholds,
    F_beta=F_beta,
    freq_ovp=freq_ovp,
    dur_factor_max=dur_factor_max,
    dur_factor_min=dur_factor_min,
    ovlp_ratio_min=ovlp_ratio_min,
    remove_duplicates=remove_duplicates,
    inherit_metadata=inherit_metadata,
    filter_deploymentID=filter_deploymentID,
    do_plot=do_plot,
)