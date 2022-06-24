# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:34:55 2022

This script creat a duirnal plot of detections or annotations

# TO DO
 - Dates/time are a bit off -> need to correct this


@author: xavier.mouy
"""

import sys
sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.

from ecosound.core.annotation import Annotation
import ecosound

## inputs #####################################################################
annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ\detections.sqlite'
annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ_run1\detections.nc'
#annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_GA_201611_CH6\detections.nc'
#annot_file = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations.nc'

integration_time = '1H'
is_binary = False
norm_max = None
threshold = 0.9
###############################################################################

# Load annotations

# # manual analysis
# annot.from_netcdf(annot_file)
# data = annot.data

# # detector 1
# annot.from_sqlite(annot_file)
# data = annot.data
# data = data[data['confidence']>=threshold]

# # # detector 2
# annot.from_netcdf(annot_file)
# data = annot.data
# data = data[data['confidence']>=threshold]

