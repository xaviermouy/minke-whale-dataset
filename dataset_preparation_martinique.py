# -*- coding: utf-8 -*-
"""
This script takes the Raven annotations made for the location "Martinique" and
convert them into the standard ecosound format.

Created on Fri Jun 10 11:33:44 2022

@author: xavier.mouy
"""
from ecosound.core.tools import list_files, filename_to_datetime
from ecosound.core.annotation import Annotation
from datetime import datetime
import uuid
import pandas as pd
import os

audio_dir = r'\\stellwagen.nefsc.noaa.gov\stellwagen\ACOUSTIC_DATA\BOTTOM_MOUNTED\NEFSC_CARIBBEAN\NEFSC_CARIBBEAN_201612_MTQ\in_water'
annot_dir = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\old_format'
out_dir = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ'
deploymentinfo_file = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\deployment_info.csv'
audio_channel = 1
label_class = 'MW'

# list all sound files with their respective  dates
audio_files = list_files(audio_dir,
                          '.aif',
                          recursive=False,
                          case_sensitive=True)
files_timestamp = filename_to_datetime(audio_files)
audio_files =pd.DataFrame({'filename': audio_files, 'timestamp':files_timestamp })

# load all annotations
annot  = Annotation()
annot_files = list_files(annot_dir,
                         '.Table.1.selections.txt',
                         recursive=False,
                         case_sensitive=True)
annot_data = annot._import_csv_files(annot_files)
annot_data['Timestamp'] = annot_data['Begin Date'] + ' ' + annot_data['Begin Clock Time']
annot_data['Timestamp'] = pd.to_datetime(annot_data['Timestamp'])

# Match annotations to audio files, then add relative times and duration
for idx, annot_row in annot_data.iterrows():
    # match annot to file
    file_match = audio_files[audio_files['timestamp']<annot_row['Timestamp']].tail(1)
    annot_tmp = Annotation()    
    annot_tmp.data['Begin Path'] = file_match['filename']
    annot_tmp.data['audio_file_start_date'] = file_match['timestamp']
    annot_tmp.data['audio_channel'] = audio_channel
    annot_tmp.data['audio_file_name'] = os.path.splitext(os.path.basename(file_match['filename'].values[0]))[0]
    annot_tmp.data['audio_file_dir'] = os.path.dirname(file_match['filename'].values[0])    
    annot_tmp.data['audio_file_extension'] = os.path.splitext(os.path.basename(file_match['filename'].values[0]))[1]    
    timedelta_start = (annot_row['Timestamp'] - file_match['timestamp']).dt.total_seconds().values[0]
    timedelta_end = timedelta_start  + annot_row['End Time (s)']-annot_row['Begin Time (s)']    
    annot_tmp.data['time_min_offset'] = timedelta_start
    annot_tmp.data['time_max_offset'] = timedelta_end
    annot_tmp.data['time_min_date'] = annot_tmp.data['audio_file_start_date'] + pd.to_timedelta(annot_tmp.data['time_min_offset'], unit='s')              
    annot_tmp.data['time_max_date'] = annot_tmp.data['audio_file_start_date'] + pd.to_timedelta(annot_tmp.data['time_max_offset'], unit='s')    
    annot_tmp.data['frequency_min'] = annot_row['Low Freq (Hz)']
    annot_tmp.data['frequency_max'] = annot_row['High Freq (Hz)']    
    annot_tmp.data['label_class'] = label_class
    annot_tmp.data['label_subclass'] = annot_row['pulse train type']
    annot_tmp.data['from_detector'] = False
    annot_tmp.data['software_name'] = 'raven'
    annot_tmp.data['uuid'] = annot_tmp.data.apply(lambda _: str(uuid.uuid4()), axis=1)
    annot_tmp.data['duration'] = annot_tmp.data['time_max_offset'] - annot_tmp.data['time_min_offset']
    annot_tmp.insert_metadata(deploymentinfo_file)    
    annot_tmp.check_integrity(verbose=False, ignore_frequency_duplicates=True)
    # stack annotations
    if idx == 0:
        annot = annot_tmp
    else:
        annot = annot + annot_tmp   
annot.check_integrity(verbose=True, ignore_frequency_duplicates=True)
print(annot)

annot.to_netcdf(os.path.join(out_dir, 'Annotations_dataset_' + annot.data['deployment_ID'][0] +' annotations.nc'))
annot.to_raven(out_dir, outfile='Annotations_dataset_' + annot.data['deployment_ID'][0] +'.Table.1.selections.txt', single_file=True)
