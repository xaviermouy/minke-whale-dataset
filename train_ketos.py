# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:45:37 2022

@author: xavier.mouy
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import ketos.data_handling.database_interface as dbi
from ketos.neural_networks.resnet import ResNetInterface
from ketos.data_handling.data_feeding import BatchGenerator

## INPUT ARGUMENTS ############################################################

# command line arguments
parser = argparse.ArgumentParser(description="Input arguments")
parser.add_argument("--db_file", type=str, action='store', required=False)
parser.add_argument("--out_dir", type=str, action='store', required=False)
parser.add_argument("--recipe_file", type=str, action='store', required=False)
parser.add_argument("--spec_config_file", type=str, action='store', required=False)
parser.add_argument("--checkpoints_dir", type=str, action='store', required=False)
parser.add_argument("--logs_dir", type=str, action='store', required=False)
parser.add_argument("--batch_size", type=int, action='store', required=False)
parser.add_argument("--n_epochs", type=int, action='store', required=False)
args = parser.parse_args()

# default values
params = dict()
params['db_file'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\ketos_databases\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800\database.h5'
params['out_dir'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\ketos_models'
params['recipe_file'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\recipe.json'
params['spec_config_file'] = r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\ketos_databases\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800\spec_config.json'
params['checkpoints_dir'] = r"C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\checkpoints"
params['logs_dir'] = r"C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\logs"
params['batch_size'] = 20 #128
params['shuffle'] = True
params['refresh_on_epoch_end_train'] = True
params['refresh_on_epoch_end_test'] = False
params['n_epochs'] = 30

# parse cli arguments if any
args = parser.parse_args()
if args.db_file:
    params['db_file'] = args.db_file
if args.out_dir:
    params['out_dir'] = args.out_dir
if args.recipe_file:
    params['recipe_file'] = args.recipe_file
if args.spec_config_file:
    params['spec_config_file'] = args.spec_config_file
if args.checkpoints_dir:
    params['checkpoints_dir'] = args.checkpoints_dir
if args.logs_dir:
    params['logs_dir'] = args.logs_dir 
if args.batch_size:
    params['batch_size'] = args.batch_size
if args.n_epochs:
    params['n_epochs'] = args.n_epochs

## ############################################################################

# init random seed to keep same results for each run
np.random.seed(1000)
tf.random.set_seed(2000)

# create output folder if it doesn't exist yet
if os.path.isdir(params['out_dir']) is False:
    os.mkdir(params['out_dir'])

# creating data feed
db = dbi.open_file(params['db_file'], 'r')
train_data = dbi.open_table(db, "/train/data")
val_data = dbi.open_table(db, "/test/data")

train_generator = BatchGenerator(batch_size=params['batch_size'],
                                 data_table=train_data,
                                 output_transform_func=ResNetInterface.transform_batch,
                                 shuffle=params['shuffle'],
                                 refresh_on_epoch_end=params['refresh_on_epoch_end_train']
                                 )

val_generator = BatchGenerator(batch_size=params['batch_size'],
                               data_table=val_data,
                               output_transform_func=ResNetInterface.transform_batch,
                               shuffle=True,
                               refresh_on_epoch_end=params['refresh_on_epoch_end_test']
                               )

# Creating neural net structure (from recipe file)
resnet = ResNetInterface.build_from_recipe_file(params['recipe_file'])
resnet.train_generator = train_generator
resnet.val_generator = val_generator
resnet.checkpoint_dir = params['checkpoints_dir']
resnet.log_dir = params['logs_dir']

# start training the network
resnet.train_loop(n_epochs=params['n_epochs'], log_csv=True, verbose=True)

db.close()
resnet.save_model(os.path.join(params['out_dir'],'ketos_model.kt'), audio_repr_file=params['spec_config_file'])
resnet.save_model(os.path.join(params['out_dir'],'ketos_model.ktpb'), audio_repr_file=params['spec_config_file'])