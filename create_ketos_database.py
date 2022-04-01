# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:00:17 2022

@author: xavier.mouy
"""

#
# # script 1:
#     - load annotation nc file
#     - define noise and target class
#     - define train and test annotations
#     - write train.csv and test.cav
#     - decimate and copy wav file sin a "train" and "test" folder
#     - define data augmentation etc
#     - write Ketos database


import pandas as pd
import ecosound
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.spectrogram import MagSpectrogram
from ketos.data_handling.parsing import load_audio_representation

annot_train = pd.read_csv("annotations_train.csv")
annot_test = pd.read_csv("annotations_test.csv")

## Input parameters #################################################################
classif_window_sec = 3.0  # segemnt duration in sec for the CNN
aug_win_step_sec = 0      # step between consecutive windows in sec (0: no augmentation)
aug_min_annot_ovlp = 0.5  # windows must contain at least 50% of the annotation
spectro_config_file = 'spec_config.json'

## ############################################################################

# reformat annotations
sl.is_standardized(annot_train)
map_to_ketos_annot_std = {'sound_file': 'filename'} 
std_annot_train = sl.standardize(table=annot_train,
                                 labels=["upcall"],
                                 mapper=map_to_ketos_annot_std,
                                 trim_table=True)
std_annot_test = sl.standardize(table=annot_test,
                                labels=["upcall"],
                                mapper=map_to_ketos_annot_std,
                                trim_table=True)


# Training set for target class: Segmentation + Augmentation
positives_train = sl.select(annotations=std_annot_train,
                            length=classif_window_sec,
                            step=aug_win_step_sec,
                            min_overlap=aug_min_annot_ovlp,
                            center=False)


# Test set for target class: Segmentation
positives_test = sl.select(annotations=std_annot_test,
                           length=classif_window_sec,
                           step=classif_window_sec,   # no overlap between frames on the test set
                           min_overlap=aug_min_annot_ovlp,
                           center=False)


# Training and test set for noise class
file_durations_train = sl.file_duration_table('data/train')
file_durations_test = sl.file_duration_table('data/test') 
negatives_train = sl.create_rndm_backgr_selections(annotations=std_annot_train,
                                                    files=file_durations_train,
                                                    length=classif_window_sec,
                                                    num=len(positives_train),
                                                    trim_table=True)

negatives_test = sl.create_rndm_backgr_selections(annotations=std_annot_train,
                                                  files=file_durations_test,
                                                  length=classif_window_sec,
                                                  num=len(positives_test),
                                                  trim_table=True)

selections_train = positives_train.append(negatives_train, sort=False)
selections_test = positives_test.append(negatives_test, sort=False)


## creatae spectrogram database
spec_cfg = load_audio_representation(spectro_config_filefig_file, name="spectrogram")

dbi.create_database(output_file='database.h5',
                    data_dir='data/train',
                    dataset_name='train',
                    selections=selections_train,
                    audio_repres=spec_cfg)


dbi.create_database(output_file='database.h5',
                    data_dir='data/test',
                    dataset_name='test',
                    selections=selections_test,
                    audio_repres=spec_cfg)

db = dbi.open_file("database.h5", 'r')

"""
Examples:
            >>> import tables
            >>> from ketos.data_handling.database_interface import open_file, create_table, table_description, table_description_annot, write
            >>> from ketos.audio.spectrogram import MagSpectrogram
            >>> from ketos.audio.waveform import Waveform
            >>>
            >>> # Create an Waveform object from a .wav file
            >>> audio = Waveform.from_wav('ketos/tests/assets/2min.wav')
            >>> # Use that signal to create a spectrogram
            >>> spec = MagSpectrogram.from_waveform(audio, window=0.2, step=0.05)
            >>> # Add a single annotation
            >>> spec.annotate(label=1, start=0., end=2.)
            >>>
            >>> # Open a connection to a new HDF5 database file
            >>> h5file = open_file("ketos/tests/assets/tmp/database2.h5", 'w')
            >>> # Create table descriptions for storing the spectrogram data
            >>> descr_data = table_description(spec)
            >>> descr_annot = table_description_annot()
            >>> # Create tables
            >>> tbl_data = create_table(h5file, "/group1/", "table_data", descr_data) 
            >>> tbl_annot = create_table(h5file, "/group1/", "table_annot", descr_annot) 
            >>> # Write spectrogram and its annotation to the tables
            >>> write(spec, tbl_data, tbl_annot)
            >>> # flush memory to ensure data is put in the tables
            >>> tbl_data.flush()
            >>> tbl_annot.flush()
            >>>
            >>> # Check that the spectrogram data have been saved 
            >>> tbl_data.nrows
            1
            >>> tbl_annot.nrows
            1
            >>> # Check annotation data
            >>> tbl_annot[0]['label']
            1
            >>> tbl_annot[0]['start']
            0.0
            >>> tbl_annot[0]['end']
            2.0
            >>> # Check audio source data
            >>> tbl_data[0]['filename'].decode()
            '2min.wav'
            >>> h5file.close()
    """
