# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:05:25 2022

This script prepares audio and annotation data for creating a ketos database

- splits the annotations into a train and test dataset
    - split ratio controlled by user
    - Stratified split between train and test
    - Group based split: Annotations coming the same deployment and same hour
      must stay together in either the train or test set (avoids background 
      matching issue during training).

- creates a train.csv and test.csv file to be used by Ketos
- moves audio files into a train and test folder
- decimates audio files based on user parameters.

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
import soundfile as sf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import scipy
import os
import csv

def decimate(train_file,train_dir,params):

    # read data
    sig, fs = sf.read(train_file, always_2d=True)
    # downsample to user-defined sampling rate
    downsampling_factor = int(np.round(fs/params['sanpling_rate_hz']))
    sig_decimated = scipy.signal.decimate(sig, downsampling_factor, n=params['filter_order'], ftype=params['filter_type'], axis=0, zero_phase=True)
    # normalize for zero mean and max = 1
    sig_decimated = sig_decimated - np.mean(sig_decimated)
    sig_decimated = sig_decimated / np.max(sig_decimated)
    # save decimated file
    outfilename = os.path.basename(os.path.splitext(train_file)[0]) + '.wav'
    sf.write(os.path.join(train_dir,outfilename), sig_decimated, int(np.round(fs/downsampling_factor)), subtype='PCM_24', endian=None, format=None, closefd=True)


params = dict()
params['dataset_file_path'] = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\Annotations_dataset_NN-HK-MW-HB_20220316T142143.nc'
params['out_dir'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke'
params['group_timeframe'] = 'H'
params['train_ratio'] = 0.75

params['sanpling_rate_hz'] = 2000
params['filter_type'] = 'iir'
params['filter_order'] = 8


# create output folders for this dataset
current_dir_name = datetime.strftime(datetime.now(),'%Y%m%dT%H%M%S')
current_out_dir = os.path.join(params['out_dir'],current_dir_name)
os.mkdir(current_out_dir)
#current_out_dir = params['out_dir']
        
# writes input parameters as csv file for book keeping purposes
log_file = open(os.path.join(current_out_dir,"dataset_parameters.csv"), "w")
writer = csv.writer(log_file) 
for key, value in params.items():
    writer.writerow([key, value])
log_file.close()

# Load dataset
dataset = Annotation()
dataset.from_netcdf(params['dataset_file_path'])
dataset.data['label'] = dataset.data['label_class'] + dataset.data['label_subclass']
data = dataset.data

# Add group ID for train/test splits (annotations with the same groupID can't be in different train or test sets)
data.insert(0,'TimeLabel',data['time_min_date'].dt.round(params['group_timeframe']).apply(lambda x: x.strftime('%Y%m%d%H%M%S')))
#data.insert(0,'group_label',data['label_class'] + '_' + data['TimeLabel'] + '_' + data['deployment_ID'])
data.insert(0,'group_label', data['TimeLabel'] + '_' + data['deployment_ID'])
labels = list(set(data['group_label']))
IDs = [*range(0,len(labels))]
data.insert(0,'group_ID', -1)
for n, label in enumerate(labels):
    data.loc[data['group_label'] == label, 'group_ID'] = IDs[n]
data.drop(columns = ['TimeLabel'])

# Add class-subclass ID for train/test stratification
data.insert(0, 'class_ID', data['label_class'] + data['label_subclass'])


# Splits data into train and test sets
n_splits = round(1/(1-params['train_ratio']))
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=None)
for train_index, test_index in skf.split(data, data['class_ID'],groups=data['group_ID']):
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    break

data_train.to_csv(os.path.join(current_out_dir,"train.csv"))
data_test.to_csv(os.path.join(current_out_dir,"test.csv"))

# create train folder and decimate audio file
train_files = data_train['audio_file_dir'] + '\\' + data_train['audio_file_name'] + data_train['audio_file_extension']
train_files = list(set(train_files))
train_dir = os.path.join(current_out_dir, 'train_data')
os.mkdir(train_dir)
for train_file in train_files:
    decimate(train_file, train_dir, params)

# create test folder and decimate audio file
test_files = data_test['audio_file_dir'] + '\\' + data_test['audio_file_name'] + data_test['audio_file_extension']
test_files = list(set(test_files))
test_dir = os.path.join(current_out_dir, 'test_data')
os.mkdir(test_dir)
for test_file in test_files:
    decimate(test_file, test_dir, params)
