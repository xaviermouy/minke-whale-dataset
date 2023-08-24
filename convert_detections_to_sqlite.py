# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 09:46:35 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement
import os

## input parameters ###########################################################
# detec_dir = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\RCA_Analysis\detector\Results_1207988255_F_SpringBay_Spring"
# out_file = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\RCA_Analysis\detector\Results_1207988255_F_SpringBay_Spring\Results_1207988255_F_SpringBay_Spring.sqlite"
###############################################################################

detec_dir = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\RCA_Analysis\detector\Detector_Fernie_nc\Detector_Fernie_nc"
out_file = r"C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\RCA_Analysis\detector\Detector_Fernie_nc\Detector_Fernie_nc\detections.sqlite"


# Load detections
detec = Measurement()
detec.from_netcdf(detec_dir)

# save as sqllite database
detec.to_sqlite(out_file)
