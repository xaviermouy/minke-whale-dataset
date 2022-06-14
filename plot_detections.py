# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:34:55 2022

This script creat a duirnal plot of detections or annotations

# TO DO
 - Dates/time are a bit off -> need to correct this


@author: xavier.mouy
"""
from ecosound.core.annotation import Annotation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc # For the legend
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime as dt
from sqlite3 import connect


# Another utility for the legend
from matplotlib.cm import ScalarMappable

def resample(data, integration_time='1H', resampler='count'):
    if resampler == 'count':
        #Lmean = data.resample(integration_time, loffset=None, label='left').apply(count)
        data_new = data.resample(integration_time, loffset=None, origin='start_day', label='left').count()
    
    data_out = pd.DataFrame({'datetime':data_new.index, 'value': data_new['uuid']})
    data_out.reset_index(drop=True,inplace=True)
    return data_out

## inputs #####################################################################
#annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ\detections.sqlite'
annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ_run1\detections.nc'
annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_GA_201611_CH6\detections.nc'
#annot_file = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations.nc'

integration_time = '1H'
is_binary = False
norm_max = None

threshold = 0.9
###############################################################################

# Load annotations
annot = Annotation()

# # manual analysis
# annot.from_netcdf(annot_file)
# data = annot.data

# # detector 1
# conn = connect(annot_file)
# annot.data = pd.read_sql_query("SELECT * FROM detections", conn)
# data = annot.data
# data = data[data['confidence']>=threshold]
# data['time_min_date'] = pd.to_datetime(data['time_min_date'], format="%Y-%m-%d %H:%M:%S") # Convert date

# detector 2
annot.from_netcdf(annot_file)
data = annot.data
data = data[data['confidence']>=threshold]
#data['time_min_date'] = pd.to_datetime(data['time_min_date'], format="%Y-%m-%d %H:%M:%S") # Convert date

# resample
data.set_index("time_min_date", inplace=True)
data_resamp = resample(data, integration_time=integration_time)

# Resample time series
data_resamp['date']=data_resamp['datetime'].dt.date
data_resamp['time']=data_resamp['datetime'].dt.time

# Create 2D matrix
axis_date = sorted(data_resamp['date'].unique())
axis_TOD = sorted(data_resamp['time'].unique()) # time of day
axis_TOD = [dt.datetime.combine(axis_date[0], x) for x in axis_TOD]
data_grid = pd.pivot_table(data_resamp, values='value', index='time',columns='date', aggfunc=np.sum)
data_grid = data_grid.fillna(0) # replaces NaNs by zeros
if is_binary:
    data_grid[data_grid>0]=1

# Plot matrix
x_lims = mdates.date2num([axis_date[0],axis_date[-1]])
y_lims = mdates.date2num([axis_TOD[0],axis_TOD[-1]])
fig, ax = plt.subplots(constrained_layout=True)
im = ax.imshow(data_grid, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], vmin=0, vmax=norm_max,aspect='auto',origin='lower',cmap='viridis')
ax.xaxis_date()
ax.yaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
#cbar = plt.colorbar(im, fraction=1, pad=0)
cbar.set_label('Detections')
fig.tight_layout()
#plt.setp(ax.get_xticklabels(), rotation = 0)
#fig.autofmt_xdate()
#ax.colorbar()
