# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:53:45 2022

This script correct labels from teh annotation dataset based on manual verification
It only needed to be used once...

@author: xavier.mouy
"""
import os
import pandas as pd
from ecosound.core.annotation import Annotation

annot_files=[
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NAVFAC-FL-200912-Site1\Annotations_dataset_USA-NAVFAC-FL-200912-Site1 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200609-CH01\Annotations_dataset_USA-NEFSC-SBNMS-200609-CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200609-CH02\Annotations_dataset_USA-NEFSC-SBNMS-200609-CH02 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200609-CH04\Annotations_dataset_USA-NEFSC-SBNMS-200609-CH04 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200809-NOPP4_CH01\Annotations_dataset_USA-NEFSC-SBNMS-200809-NOPP4_CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200809-NOPP4_CH09\Annotations_dataset_USA-NEFSC-SBNMS-200809-NOPP4_CH09 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200809-NOPP4_CH10\Annotations_dataset_USA-NEFSC-SBNMS-200809-NOPP4_CH10 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200812-NOPP5_CH10\noise\Annotations_dataset_USA-NEFSC-SBNMS-200812-NOPP5_CH10 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200903-NOPP6_CH03\Annotations_dataset_USA-NEFSC-SBNMS-200903-NOPP6_CH03 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200903-NOPP6a_CH01\Annotations_dataset_USA-NEFSC-SBNMS-200903-NOPP6a_CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200903-NOPP6a_CH01\noise\Annotations_dataset_USA-NEFSC-SBNMS-200903-NOPP6a_CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200905-NOPP7a_CH01\Annotations_dataset_USA-NEFSC-SBNMS-200905-NOPP7a_CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200907-NOPP8a_CH01\Annotations_dataset_USA-NEFSC-SBNMS-200907-NOPP8a_CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200910-NOPP8_CH01\Annotations_dataset_USA-NEFSC-SBNMS-200910-NOPP8_CH01 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200910-NOPP8_CH03\Annotations_dataset_USA-NEFSC-SBNMS-200910-NOPP8_CH03 annotations',
    r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\USA-NEFSC-SBNMS-200912-NOPP9_CH10\noise\Annotations_dataset_USA-NEFSC-SBNMS-200912-NOPP9_CH10 annotations',
    ]

# manual label correction
HB_label_ID = pd.read_csv(r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\Annotations_dataset_MW-NN_20220204T192254_to_label_as_HB.txt',header=None)
FS_label_ID = pd.read_csv(r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\Annotations_dataset_MW-NN_20220204T192254_to_label_as_FS.txt',header=None)
NN_label_ID = pd.read_csv(r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\Annotations_dataset_MW-NN_20220204T192254_to_label_as_NN.txt',header=None)
delete_label_ID = pd.read_csv(r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\Annotations_dataset_MW-NN_20220204T192254_to_delete.txt',header=None)
HB_label_ID = list(HB_label_ID[0].values)
FS_label_ID = list(FS_label_ID[0].values)
NN_label_ID = list(NN_label_ID[0].values)
delete_label_ID = list(delete_label_ID[0].values)

# Do the changes
for annot_file in annot_files:
    print(annot_file)
    # load annotations
    an = Annotation()
    an.from_netcdf(annot_file + '.nc')
    data = an.data
    # make the changes
    for HB_ID in HB_label_ID:
        data['label_class'].loc[data['uuid'] == HB_ID[0:-4]]='HB' 
    for FS_ID in FS_label_ID:
        data['label_class'].loc[data['uuid'] == FS_ID[0:-4]]='FS'
    for NN_ID in NN_label_ID:
            data['label_class'].loc[data['uuid'] == NN_ID[0:-4]]='NN'
    for delete_ID in delete_label_ID:
        data = data.drop(data[data['uuid'] == delete_ID[0:-4]].index)
    # save new files
    an.data = data
    an.to_netcdf(annot_file + '_corrected.nc')


# double check that everything was changed correctly
idx=0
for annot_file in annot_files:
    print(annot_file)
    # load annotations
    if idx==0:
        an2 = Annotation()
        an2.from_netcdf(annot_file + '_corrected.nc')
    else:
        an3 = Annotation()
        an3.from_netcdf(annot_file + '_corrected.nc')
        an2 = an2 + an3
    idx+=1
print(an2.summary())
    

