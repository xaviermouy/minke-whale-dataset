# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 10:16:31 2022

@author: xavier.mouy
"""
import datetime
from ecosound.core.annotation import Annotation
from ecosound.evaluation.prf import PRF
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

###############################################################################
##  input parameters ##########################################################
###############################################################################


# # Annotation file:
# annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations_withSNR.nc"
# # Detection folder or detection sqllite file
# detec_file =r'D:\NOAA\2022_Minke_whale_detector\results\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\FRA-NEFSC-CARIBBEAN-201612-MTQ\wrong\detections.sqlite'
# date_min = datetime.datetime(year=2016, month=12, day=26)
# date_max = datetime.datetime(year=2017, month=2, day=26)
# # Name of the class to test
# target_class = "MW"
# threshold =0.6

# # Annotation file:
# annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc"
# # Detection folder or detection sqllite file
# detec_file =r'D:\NOAA\2022_Minke_whale_detector\results\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\UK-SAMS-N1\detections.sqlite'
# date_min = datetime.datetime(year=2020, month=11, day=2)
# date_max = datetime.datetime(year=2020, month=11, day=16)
# # Name of the class to test
# target_class = "MW"
# threshold =0.6

# # Annotation file:
# annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc"
# # Detection folder or detection sqllite file
# detec_file =r'D:\NOAA\2022_Minke_whale_detector\results\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\UK-SAMS-N1\detections.sqlite'
# date_min = datetime.datetime(year=2020, month=12, day=28)
# date_max = datetime.datetime(year=2021, month=1, day=11)
# # Name of the class to test
# target_class = "MW"
# threshold =0.6

# Annotation file:
annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\USA-NEFSC-GA-201612-CH6\Annotations_dataset_USA-NEFSC-GA-201611-CH6 annotations_withSNR.nc"
# Detection folder or detection sqllite file
detec_file =r'D:\NOAA\2022_Minke_whale_detector\results\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm\USA-NEFSC-GA-201612-CH6\detections.sqlite'
date_min = datetime.datetime(year=2016, month=12, day=17)
date_max = datetime.datetime(year=2016, month=12, day=28)
# Name of the class to test
target_class = "MW"
threshold =0.6

###############################################################################
###############################################################################

# load ground truth data
print(" ")
print("Loading manual annotations...")
annot = Annotation()
annot.from_netcdf(annot_file)
print("Annotation labels:")
print(annot.get_labels_class())

# load destections
print(" ")
print("Loading automatic detections...")
detec = Annotation()
detec.from_sqlite(detec_file)


# filter to given dates
detec.data.query('time_min_date >= @date_min and time_max_date < @date_max', inplace=True)
annot.data.query('time_min_date >= @date_min and time_max_date < @date_max', inplace=True)


#annot.heatmap(title='Annotations',integration_time='1D',is_binary=True,colormap='gray_r') #norm_value=40
detec.heatmap(title='Detections',integration_time='1D',is_binary=True, colormap='gray_r')
detec.heatmap(title='Detections',integration_time='1D',is_binary=False, colormap='gray_r')

#annot_TS = annot.calc_time_aggregate_1D(integration_time='1D',is_binary=True)
#detec_TS = detec.calc_time_aggregate_1D(integration_time='1D',is_binary=True)
#plt.plot(annot_TS)
#plt.plot(detec_TS)

plt.show()
print ('done')