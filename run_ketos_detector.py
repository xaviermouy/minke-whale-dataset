# ================================================================================ #
#   Authors: Fabio Frazao and Oliver Kirsebom                                      #
#   Contact: fsfrazao@dal.ca, oliver.kirsebom@dal.ca                               #
#   Organization: MERIDIAN (https://meridian.cs.dal.ca/)                           #
#   Team: Data Analytics                                                           #
#   Project: ketos                                                                 #
#   Project goal: The ketos library provides functionalities for handling          #
#   and processing acoustic data and applying deep neural networks to sound        #
#   detection and classification tasks.                                            #
#                                                                                  #
#   License: GNU GPLv3                                                             #
#                                                                                  #
#       This program is free software: you can redistribute it and/or modify       #
#       it under the terms of the GNU General Public License as published by       #
#       the Free Software Foundation, either version 3 of the License, or          #
#       (at your option) any later version.                                        #
#                                                                                  #
#       This program is distributed in the hope that it will be useful,            #
#       but WITHOUT ANY WARRANTY; without even the implied warranty of             #
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              #
#       GNU General Public License for more details.                               # 
#                                                                                  #
#       You should have received a copy of the GNU General Public License          #
#       along with this program.  If not, see <https://www.gnu.org/licenses/>.     #
# ================================================================================ #

""" This scripts transforms a Ketos (binary) classifier model into a detector capable of 
    processing continuous audio data and outputting a list of detections.

    The audio data is processed by sliding a window across the audio data, 
    using a user-specified step size. At each step, the audio segment contained 
    within the window is passed on to the model for classification.

    A number of options are available for modifying the behaviour of the detector, 
    including post-processing of the classification scores reported by the model.

    The script is executed in the terminal with the following command,

        python detector.py --model <path_to_saved_model> --audio_folder <path_to_data_folder>
    
    To see the full list of command lines arguments, type

        python detector.py --help
"""

import os
import argparse
from tqdm import tqdm
import pandas as pd
import ketos.neural_networks.dev_utils.detection as det
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.resnet import ResNetInterface
import ketos.neural_networks
from ketos.neural_networks.dev_utils.detection import process, save_detections, merge_overlapping_detections
from ecosound.core.annotation import Annotation
import os
import soundfile as sf
import ecosound.core.tools
import numpy as np
import scipy
from datetime import datetime
import uuid
import platform
import logging
import time

def set_logger(outdir):
    """
    Set up the logs.
    Configure the error and info logs
    Parameters
    ----------
    outdir : str
        Path of the folder where the logs will be saved.
    Returns
    -------
    logger : logger object
        Allows to add error or info to the logs.
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Create debug logger
    info_handler = logging.FileHandler(os.path.join(outdir, 'full_log.txt'))
    info_handler.setLevel(logging.DEBUG)
    info_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    info_handler.setFormatter(info_format)
    logger.addHandler(info_handler)
    # Create error logger
    error_handler = logging.FileHandler(os.path.join(outdir, 'errors_log.txt'))
    error_handler.setLevel(logging.ERROR)
    error_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    error_handler.setFormatter(error_format)
    logger.addHandler(error_handler)
    return logger

def decimate(infile,out_dir,sampling_rate_hz, filter_order=8, filter_type='iir', channel=1):

    # read data
    sig, fs = sf.read(infile, always_2d=True)
    # select channel
    sig = sig[:, channel-1]    
    # downsample to user-defined sampling rate
    downsampling_factor = int(np.round(fs/sampling_rate_hz))
    sig_decimated = scipy.signal.decimate(sig, downsampling_factor, n=filter_order, ftype=filter_type, axis=0, zero_phase=True)
    # normalize for zero mean and max = 1
    sig_decimated = sig_decimated - np.mean(sig_decimated)
    sig_decimated = sig_decimated / np.max(sig_decimated)
    # save decimated file
    outfilename = os.path.basename(os.path.splitext(infile)[0]) + '.wav'
    sf.write(os.path.join(out_dir,outfilename), sig_decimated, int(np.round(fs/downsampling_factor)), subtype='PCM_24', endian=None, format=None, closefd=True)

    return outfilename

parser = argparse.ArgumentParser(description="Ketos acoustic signal detection script")

# define command line arguments
parser.add_argument('--model', type=str, default=None,
                        help='path to the trained ketos classifier model')
parser.add_argument('--audio_folder', type=str, default=None,
                        help='path to the folder containing the .wav files')
parser.add_argument('--channel', type=int, default=1,
                        help='Audio channel to use. Default is 1.')
parser.add_argument('--extension', type=str, default='.wav',
                        help='Extension of audio files to process. Default is ".wav".')  
parser.add_argument('--output_folder', type=str, default='detections.csv',
                        help='the .csv file where the detections will be saved. An existing file will be overwritten.')
parser.add_argument('--num_segs', type=int, default=128,
                        help='the number of segment to hold in memory at one time')
parser.add_argument('--step_size', type=float, default=None,
                        help='step size (in seconds) used for the sliding window')
parser.add_argument('--buffer', type=float, default=0.0,
                        help='Time (in seconds) to be added on either side of every detected signal')
parser.add_argument('--win_len', type=int, default=1,
                        help='Length of score averaging window (no. time steps). Must be an odd integer.')
parser.add_argument('--threshold', type=float, default=0.5,
                        help='minimum score for a detection to be accepted (ranging from 0 to 1)')                   
parser.add_argument('--duration', type=float, default=5,
                        help='Duration of classification window in seconds')  
parser.add_argument('--tmp_dir', type=str, default='./tmp',
                        help='Path of temporary folder.')  
parser.add_argument('--deployment_file', type=str, default=None,
                        help='deployment_info.csv with metadata.')  

show_progress_parser = parser.add_mutually_exclusive_group(required=False)
show_progress_parser.add_argument('--show_progress', dest='progress_bar', action='store_true')
show_progress_parser.add_argument('--hide_progress', dest='progress_bar', action='store_false')

group_parser = parser.add_mutually_exclusive_group(required=False)
group_parser.add_argument('--with_group', dest='group', action='store_true')
group_parser.add_argument('--without_group', dest='group', action='store_false')

group_parser = parser.add_mutually_exclusive_group(required=False)
group_parser.add_argument('--with_merge', dest='merge', action='store_true')
group_parser.add_argument('--without_merge', dest='merge', action='store_false')

parser.set_defaults(progress_bar=False, group=False, merge=False) 

# parse command line args
args = parser.parse_args()
assert isinstance(args.win_len, int) and args.win_len%2 == 1, 'win_len must be an odd integer'


###############################################################################

recursive=True

# creates temp and output folder if not alreadey there
if os.path.isdir(args.tmp_dir) is False:
    os.mkdir(args.tmp_dir)
if os.path.isdir(args.output_folder) is False:
    os.mkdir(args.output_folder)

# Set error logs
logger = set_logger(args.output_folder)

# load the classifier and the spectrogram parameters
#model1, audio_repr1 = ResNetInterface.load_model_file(args.model, './tmp', load_audio_repr=True)
model, audio_repr = ketos.neural_networks.load_model_file(args.model, args.tmp_dir, load_audio_repr=True)
spec_config = audio_repr[0]['spectrogram']
spec_config['duration']=args.duration

# list files to process
if os.path.isfile(args.audio_folder):  # if a single file was provided
    files = [args.audio_folder]
elif os.path.isdir(args.audio_folder):  # if a folder was provided
    files = ecosound.core.tools.list_files(args.audio_folder,
                                            args.extension,
                                            recursive=recursive,
                                            case_sensitive=True)
logger.info('Files to process: ' + str(len(files)))
start_time_loop = time.time()
# loop to process each file
for idx,  file in enumerate(files):
    try:
        
        logger.info(file)
        start_time = time.time()
        print(str(idx+1) + r'/' + str(len(files)) + ': ' + file)
      
        # Decimate
        temp_file_name = decimate(file, args.tmp_dir, spec_config['rate'], channel=args.channel)      
        # initialize the audio loader
        audio_loader = AudioFrameLoader(duration=spec_config['duration'], step=args.step_size, path=args.tmp_dir, filename=[temp_file_name], repres=spec_config)
        # process the audio data
        detections = process(provider=audio_loader, model=model, batch_size=args.num_segs, buffer=args.buffer, threshold=args.threshold, group=args.group, win_len=args.win_len, progress_bar=args.progress_bar)
        # merge overlapping detections
        if args.merge == True:
            detections = merge_overlapping_detections(detections)
        print(len(detections), ' detections')
        logger.info(" %u detections" % (len(detections)))
        
        # Save as Raven table
        if len(detections) > 0: # only if there are detections
            annot = Annotation()
            annot_data = annot.data
            files_list = [os.path.splitext(list(file)[0])[0] for file in detections]
            file_timestamp = ecosound.core.tools.filename_to_datetime(file)[0]
            
            #ext_list = [os.path.splitext(list(file)[0])[1] for file in detections]
            start_list = [list(file)[1] for file in detections]
            duration_list = [list(file)[2] for file in detections]
            confidence_list = [list(file)[3] for file in detections]
            confidence_list = [ '%.3f' % elem for elem in confidence_list ]
            end_list = [start_list[i] + duration_list[i] for i in range(len(start_list))]
            annot_data['time_max_offset'] = end_list
            annot_data['time_min_offset'] = start_list
            annot_data['frequency_min'] = spec_config['freq_min']
            annot_data['frequency_max'] = spec_config['freq_max']
            annot_data['audio_file_dir'] = args.audio_folder
            annot_data['audio_file_name'] = files_list
            annot_data['audio_file_extension'] = args.extension        
            annot_data['label_class']='MW'
            annot_data['confidence']= confidence_list
            annot_data['software_name']= 'Ketos-Minke'
            annot_data['entry_date'] = datetime.now()
            annot_data['audio_file_start_date'] = file_timestamp
            annot_data['time_min_date'] = pd.to_datetime(file_timestamp + pd.to_timedelta(annot_data['time_min_offset'], unit='s'))
            annot_data['time_max_date'] = pd.to_datetime(file_timestamp + pd.to_timedelta(annot_data['time_max_offset'], unit='s'))
            annot_data['from_detector'] = True
            annot_data['duration'] = annot_data['time_max_offset'] - annot_data['time_min_offset']
            annot_data['uuid'] = annot_data.apply(lambda _: str(uuid.uuid4()), axis=1)
            annot_data['operator_name'] = platform.uname().node        
            annot.data = annot_data
            # insert metadata
            if args.deployment_file:
                annot.insert_metadata(args.deployment_file)
            annot.insert_values(audio_channel = args.channel)
            
            #annot_data['audio_channel'] = args.channel
            # sort chronologically
            annot.data.sort_values('time_min_offset',
                                    ascending=True,
                                    inplace=True,
                                    ignore_index=True,
                                    )      
            annot.check_integrity()
            annot.to_raven(outdir=args.output_folder,single_file=False)      
            annot.to_netcdf(os.path.join(args.output_folder,files_list[0]))     
        
        proc_time_file = time.time() - start_time
        logger.info("--- Executed in %0.4f seconds ---" % (proc_time_file))
        print(f"Executed in {proc_time_file:0.4f} seconds")
        
        # delete temporary file
        os.remove(os.path.join(args.tmp_dir,temp_file_name))
          
    except BaseException as e:
      logger.error(file)
      logger.error("Exception occurred", exc_info=True)
      print('An error occured: ' + str(e))

# Wrap-up logs
proc_time_process = time.time() - start_time_loop
logger.info('Process complete.')
logger.info("--- All files processed in %0.4f seconds ---" % (proc_time_process))
print(f"All files processed in {proc_time_process:0.4f} seconds")
logging.shutdown()