import os
import time

import pathlib
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import resample

HOME = os.path.expanduser('~')
GENDER_DICT = {
    0: 'M',
    1: 'F'
}

config = {
    'source_folder_shhs': HOME + '/Datasets/shhs/polysomnography/edfs/shhs2/',
    'source_folder_mesa': HOME + '/Datasets/mesa/polysomnography/edfs/',
    'meta_path_shhs': HOME + '/Datasets/shhs/shhs2-dataset-0.14.0.csv',
    'meta_path_mesa': HOME + '/Datasets/mesa/mesa-sleep-dataset-0.3.0.csv',

    'target_folder': 'sleep_dataset/',

    'segment_length': 32,                   # Length of extracted segments in seconds
    'num_segs_per_patient': 20,             # Number of segments to extract from one patient
    'new_fs': 256,                          # New resampled frequency

    'plot_output': False,                   # Plot the extracted segments
    'save': True,                           # Save the extracted segments

    'log_interval': 10,
}

t_start = time.time()

# Load shhs meta data
meta_shhs = pd.read_csv(config['meta_path_shhs'], encoding="ISO-8859-1")
shhs_ids = meta_shhs['nsrrid']
shhs_genders = meta_shhs['gender'] - 1

# Load mesa meta data
meta_mesa = pd.read_csv(config['meta_path_mesa'], encoding="ISO-8859-1")
mesa_ids = meta_mesa['mesaid']
mesa_genders = 1 - meta_mesa['gender1']

genders = []
keys = []

# Create paths
source_folder_shhs = config['source_folder_shhs']
source_folder_mesa = config['source_folder_mesa']
target_folder = config['target_folder']
os.makedirs(target_folder, exist_ok=True)

seg_len = config['segment_length']
new_fs = config['new_fs']
plot_output = config['plot_output']
save = config['save']

# Get all file-paths
shhs_root = pathlib.Path(source_folder_shhs)
shhs_files = [str(p) for p in list(shhs_root.glob('*/'))
              if str(p).split('/')[-1] != '.DS_Store']

mesa_root = pathlib.Path(source_folder_mesa)
mesa_files = [str(p) for p in list(mesa_root.glob('*/'))
              if str(p).split('/')[-1] != '.DS_Store']

n_files_total = len(shhs_files) + len(mesa_files)

print("# shhs {}, # mesa {}".format(len(shhs_files), len(mesa_files)))


def convert(files, ids, genders, ecg_channel_name, dataset_name, i_patient):
    """
    Extracts segments from the ECG recordings and saves them as .npy files
    The segments of every patient will be stored in a separate folder.

    args:
        files (list of str): list of paths to the .edf recordings
        ids (pd.Dataframe): internal patient ids
        genders (pd.Dataframe): gender of each patient
        ecg_channel_name (str) name of the channel with ECG-data in the .edf files
        dataset_name (str): name of dataset ('shhs' or 'mesa')
        i_patient (int): counter
    """
    for file in files:
        i_patient += 1

        # Load index of file in meta-data
        if dataset_name == 'shhs':
            meta_idx = int(np.argwhere(ids.values == int(file.split('-')[1][:-4])))
        elif dataset_name == 'mesa':
            meta_idx = int(np.argwhere(ids.values == int(file.split('-')[-1][:-4])))
        else:
            raise Exception('Invalid dataset name. Select "shhs" or "mesa".')

        # Get gender of patient from meta-data
        gender = GENDER_DICT[genders.values[meta_idx]]
        patient_id = str(ids.values[meta_idx]).zfill(6)

        # Create new folder for the patient
        patient_folder = os.path.join(
            target_folder, '{}-{}-{}'.format(dataset_name, patient_id, gender))
        os.makedirs(patient_folder, exist_ok=True)

        # Load .edf file
        with pyedflib.EdfReader(file) as f:
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()

            # Select ECG channel
            for i_channel in range(n):
                if signal_labels[i_channel] == ecg_channel_name:
                    channel = i_channel

            # Load ECG channel
            signal = f.readSignal(channel)

            # Get sample frequency
            fs = f.getSampleFrequency(channel)

            # Set start 60 Minutes in the recording to avoid noise in the beginning
            start = 60 * 60 * fs
            # use only inner 5 hours where signal is cleaner
            stop = min(len(signal) - seg_len * fs - 60 * 60 * fs, 6 * 60 * 60 * fs)
            if stop < start:
                print("WARNING: stop < start for file {}".format(file))
                continue

            seg_indices = np.linspace(
                start, stop, num=config['num_segs_per_patient'], dtype=np.int)

            for i_segment, ind in enumerate(seg_indices):
                local_idx = str(i_segment).zfill(2)
                target_path = os.path.join(
                    patient_folder, '{}.npy'.format(local_idx))

                segment = signal[ind:ind + seg_len * fs]

                if plot_output:
                    plt.title("{}/{}, time: {:.2f}h/{:.2f}h".format(
                        i_patient, n_files_total, ind / (fs * 3600),
                        len(signal) / (fs * 3600))
                    )
                    plt.plot(segment)
                    plt.show()

                # Resample
                if fs != config['new_fs']:
                    if abs(fs - config['new_fs']) > 20:
                        print("Warning, this is a big resampling, fs: {}, source: {}, target: {}".format(
                            fs, file, target_path
                        ))
                    num = int(seg_len * new_fs)
                    segment = resample(segment, num=num)

                # Save
                if save:
                    np.save(file=target_path, arr=segment)

            # Logging
            if i_patient % config['log_interval'] == 0:
                time_left = (((time.time() - t_start) / i_patient) * (n_files_total - i_patient)) / 3600
                print("\nProcessed file {} of {}".format(i_patient, n_files_total))
                print(target_path, fs, len(segment),
                      "{}/{}".format(i_patient, n_files_total),
                      "\nTime elapsed {:.2f}s".format(time.time() - t_start),
                      "Estimated time left: {:.2f} hours".format(time_left))

    return i_patient


# Convert files
i_patient = 0
i_patient = convert(shhs_files, shhs_ids, shhs_genders, 'ECG', 'shhs', i_patient)
i_patient = convert(mesa_files, mesa_ids, mesa_genders, 'EKG', 'mesa', i_patient)
