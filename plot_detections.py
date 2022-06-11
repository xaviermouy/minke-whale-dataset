# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:34:55 2022

This script creat a duirnal plot of detections or annotations

@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc # For the legend
# Another utility for the legend
from matplotlib.cm import ScalarMappable

def resample(data, integration_time='1H', resampler='count'):
    if resampler == 'count':
        #Lmean = data.resample(integration_time, loffset=None, label='left').apply(count)
        data_new = data.resample(integration_time, loffset=None, origin='start_day', label='left').count()
    
    data_out = pd.DataFrame({'datetime':data_new.index, 'value': data_new['uuid']})
    data_out.reset_index(drop=True,inplace=True)
    return data_out

annot_file = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations.nc'
integration_time = '1H'

# Load annotations
annot = Annotation()
annot.from_netcdf(annot_file)
data = annot.data

# resample
data.set_index("time_min_date", inplace=True)
data_resamp = resample(data, integration_time=integration_time)


# Reshape
data_resamp['date']=data_resamp['datetime'].dt.date
data_resamp['time']=data_resamp['datetime'].dt.time
axis_date = sorted(data_resamp['date'].unique())
axis_time = sorted(data_resamp['time'].unique())


# -> add x and y axis indices for each value of data_resamp (add x_idx, y_idx columns)
# -> loop through data_resamp and distribute into empty matrix (start matrix with Nans)
# -> plot matrix

#gg=data_resamp.pivot_table('date','time')

# # Extract hour, day, and temperature
# hour = data_resamp["date"].dt.hour
# day = data_resamp["date"].dt.day
# month = data_resamp["date"].dt.month
# year = data_resamp["date"].dt.year
# value = data_resamp["value"]

# # # Re-arrange temperature values
# #values = value.values.reshape(24, len(day.unique()), order="F")
# values = value.values.reshape(24, len(day), order="F")
# # Compute x and y grids, passed to `ax.pcolormesh()`.

# # The first + 1 increases the length
# # The outer + 1 ensures days start at 1, and not at 0.
# xgrid = np.arange(day.max() + 1) + 1

# # Hours start at 0, length 2
# ygrid = np.arange(25)

# fig, ax = plt.subplots()
# ax.pcolormesh(xgrid, ygrid, temp)
# ax.set_frame_on(False) # remove all spines