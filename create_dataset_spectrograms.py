# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:48 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.audiotools import Sound
import pandas as pd
from datetime import datetime
import os
import librosa
import librosa.display
import soundfile
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def calc_spectro(y, sr, n_fft, hop_length, win_length, title="Power spectrogram"):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        S, hop_length=hop_length, y_axis="linear", x_axis="time", sr=sr, ax=ax
    )
    # img = librosa.display.specshow(S, hop_length=hop_length,y_axis='linear', x_axis='time', sr=sr,ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig, ax, S

params = dict()
params["dataset_file_path"] = r"C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\Annotations_dataset_MW-NN_20220204T192254.nc"
params["out_dir"] = r"C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\dataset_spectrograms"
params["audio_dir"] = r"C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets"
params["sanpling_rate_hz"] = 2000
params["clips_buffer_s"] = 0
params["spectro_on"] = True
params["spetro_nfft"] = 512
params["spetro_frame"] = 512
params["spetro_inc"] = 128
params["spetro_on_npy"] = True

# Load dataset
dataset = Annotation()
dataset.from_netcdf(params["dataset_file_path"])

# define the different class names and create separate folders
outdirname = os.path.join(params["out_dir"],os.path.splitext(os.path.basename(params["dataset_file_path"]))[0])
if os.path.isdir(outdirname) == False:
    os.mkdir(outdirname)
labels = list(set(dataset.data['label_class']))

# loop through each class_labels
for label in labels:
    print(label)
    current_dir = os.path.join(outdirname, label)
    if os.path.isdir(current_dir) == False:
        os.mkdir(current_dir)
    annot_sp = dataset.data[dataset.data["label_class"] == label]
    # loop through is annot for that class label
    for idx, annot in annot_sp.iterrows():
        F = str(annot.uuid) + '.png'
        if os.path.isfile(os.path.join(current_dir, F)) == False: # only if file doesn't exist already
            print('Processing file', F)
            # define start/stop times +/- buffer
            t1 = annot.time_min_offset - params["clips_buffer_s"]
            if t1 <= 0:
                t1 = 0
            t2 = annot.time_max_offset + params["clips_buffer_s"]
            duration = t2 - t1
            # load sound clip
            y, s = librosa.load(
                os.path.join(annot["audio_file_dir"], annot["audio_file_name"])
                + annot["audio_file_extension"],
                sr=params["sanpling_rate_hz"],
                offset=t1,
                duration=duration,
            )  # Downsample 44.1kHz to 8kHz
            t2 = t1 + (len(y)/s)  # readjust end time in case it exceeded the end  of the file (which librosa handles by taking the last sample)
    
            # Create audio clip standard name        
            #title = annot.deployment_ID + '   ' + annot.audio_file_name
            title = annot.audio_file_name
            # write spectrogram image
            if params["spectro_on"]:
                fig, ax, S = calc_spectro(
                    y,
                    s,
                    params["spetro_nfft"],
                    params["spetro_inc"],
                    win_length=params["spetro_frame"],
                    title = title
                )
                fig.savefig(os.path.join(current_dir, F), dpi=600)
                plt.close("all")
                plt.close()
                plt.cla()
                plt.clf()
                #if params["spetro_on_npy"]:
                #    np.save(os.path.splitext(outfilename)[0] + ".npy", S)
            #annot_unique_id += 1
