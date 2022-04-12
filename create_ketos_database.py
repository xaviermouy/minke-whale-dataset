# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:00:17 2022

This script takes the annotation logs for the train and test sets and creates
a database of spectrograms that will be used to train/test the neural net

@author: xavier.mouy
"""


import os
import csv
import pandas as pd
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.spectrogram import MagSpectrogram


## Input parameters #################################################################
params=dict()
params['train_annot_file'] =r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\train.csv'
params['test_annot_file'] =r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\test.csv'

params['data_train_dir'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\train_data'
params['data_test_dir'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\test_data'
params['out_dir'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308'

params['positive_labels'] = ['MW']
params['negative_labels'] = ['HK', 'NN', 'HKPT', 'HKP', 'NNS', 'HB']

params['classif_window_sec'] = 5.0  # segemnt duration in sec for the CNN
params['aug_win_step_sec'] = 3.0    # step between consecutive windows in sec (0: no augmentation)
params['aug_min_annot_ovlp'] = 0.75 # windows must contain at least x% of the annotation
params['spectro_config_file'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\spec_config.json'


## ############################################################################

# load spectro_config_file
spec_cfg = load_audio_representation(params['spectro_config_file'], name="spectrogram")
if params['classif_window_sec'] != spec_cfg['duration']:
    raise Exception("classif_window_sec value does not match the one in the spectro_config_file.")
params['spec_cfg'] = spec_cfg

current_out_dir = params['out_dir']

# load annotations
annot_train = pd.read_csv(params['train_annot_file'])
annot_test = pd.read_csv(params['test_annot_file'])

# display classes available
print('Train set classes:', set(annot_train['class_ID']))
print('Test set classes:', set(annot_test['class_ID']))

# reformating file names
#annot_train['sound_file'] = annot_train['audio_file_name'] + annot_train['audio_file_extension']
#annot_test['sound_file'] = annot_test['audio_file_name'] + annot_test['audio_file_extension']
annot_train['sound_file'] = annot_train['audio_file_name'] + '.wav'
annot_test['sound_file'] = annot_test['audio_file_name'] + '.wav'

# Assigning positive and negative class labels
positive_pattern = '|'.join(params['positive_labels'])
negative_pattern = '|'.join(params['negative_labels'])
annot_train['label'][annot_train.class_ID.str.contains(positive_pattern)]=1
annot_train['label'][annot_train.class_ID.str.contains(negative_pattern)]=0
print('Train set:')
print('- Positive labels: ', sum(annot_train['label']==1))
print('- Negative labels: ', sum(annot_train['label']==0))
annot_test['label'][annot_test.class_ID.str.contains(positive_pattern)]=1
annot_test['label'][annot_test.class_ID.str.contains(negative_pattern)]=0
print('Test set:')
print('- Positive labels: ', sum(annot_test['label']==1))
print('- Negative labels: ', sum(annot_test['label']==0))

# reformat annotations to Ketos format
sl.is_standardized(annot_train)
map_to_ketos_annot_std = {'sound_file': 'filename',
                          'time_min_offset':'start',
                          'time_max_offset' :'end',
                          'frequency_min': 'min_freq',
                          'frequency_max': 'max_freq',
                          } 
std_annot_train = sl.standardize(table=annot_train,
                                 mapper=map_to_ketos_annot_std,
                                 trim_table=True)
std_annot_test = sl.standardize(table=annot_test,
                                mapper=map_to_ketos_annot_std,
                                trim_table=True)

# Training set for target class: Segmentation + Augmentation
std_annot_train_aug = sl.select(annotations=std_annot_train,
                            length=params['classif_window_sec'],
                            step=params['aug_win_step_sec'],
                            min_overlap=params['aug_min_annot_ovlp'],
                            center=True)
std_annot_train_aug.to_csv(os.path.join(current_out_dir,'train_ketos.csv'))

# Test set for target class: Segmentation
std_annot_test_aug = sl.select(annotations=std_annot_test,
                           length=params['classif_window_sec'],
                           step=params['aug_win_step_sec'],
                           min_overlap=params['aug_min_annot_ovlp'],
                           center=True)
std_annot_test_aug.to_csv(os.path.join(current_out_dir,'test_ketos.csv'))


## creatae spectrogram database
db_file = os.path.join(current_out_dir,'database.h5')
dbi.create_database(output_file=db_file,
                    data_dir=params['data_train_dir'],
                    dataset_name='train',
                    selections=std_annot_train_aug,
                    audio_repres=spec_cfg)

dbi.create_database(output_file=db_file,
                    data_dir=params['data_test_dir'],
                    dataset_name='test',
                    selections=std_annot_test_aug,
                    audio_repres=spec_cfg)

#db = dbi.open_file("database.h5", 'r')


## writes input parameters as csv file for book keeping purposes
log_file = open(os.path.join(current_out_dir,"ketos_database_parameters.csv"), "w")
writer = csv.writer(log_file) 
for key, value in params.items():
    writer.writerow([key, value])
log_file.close()