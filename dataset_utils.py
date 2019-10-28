import pathlib
import random
import torch

import numpy as np

from random import randint
from torch.utils.data.dataset import Dataset

from preprocessing import resample_frequency, obspy_highpass, notch_filter

GENDER_DICT = {
    'M': 0,
    'F': 1
}


def get_data_paths(path):
    root_dir = pathlib.Path(path)
    data_paths = [p for p in list(root_dir.glob('*/'))
                  if str(p).split('/')[-1] != '.DS_Store']
    data_paths = [[str(p) for p in list(path.glob('*'))] for path in data_paths]
    return data_paths


class ECGDataset(Dataset):
    def __init__(
            self,
            data_paths,
            is_test=False,
            fs=256,
            seg_length=16,
            high_pass=True,
            notch=True,
            instance_normalization=True,
            do_overfitting=False,
            intensity_range=None,
    ):
        """
        Dataset class to load ecg data from .npy files

        args:
            data_path (list of list of str): list strings with file paths. Every
                                             entry is a list of all paths for
                                             one patient
            is_test (bool): indicates of dataset is used for testing, which
                            means all data paths need to be considered in one
                            epoch
            fs (int): frequency of the recorded data in Hertz
            seg_length (float): length of the loaded ecg segments
            high_pass (bool): preprocess signal with high pass filter
            notch (bool): preprocess signal with notch filter
            instance_normalization (bool): normalize every signal to zero mean
                                           and unit variance
            do_overfitting (bool): overfit on first sample
            intensity_range (list of int with len 2): min and max scaling
                                                      ranges for adding noise
                                                      (e.g. [0.9, 1.1])
        """
        self.is_test = is_test
        self.fs = fs
        self.seg_length = seg_length
        self.high_pass = high_pass
        self.notch = notch
        self.instance_normalization = instance_normalization
        self.do_overfitting = do_overfitting
        self.intensity_range = intensity_range
        self.dataset_fs = 256

        if len(data_paths) == 0:
            raise (RuntimeError("Got empty list of data_paths"))

        if is_test:
            data_paths = [[p] for path in data_paths for p in path]

        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # Select random segment
        segments = self.data_paths[index]
        segment_idx = 0 if self.do_overfitting else randint(0, len(segments) - 1)

        # Load segment
        signal = np.load(segments[segment_idx]).reshape(-1, 1)

        # Get gender from filename
        gender = GENDER_DICT[
            str(segments[segment_idx].split('/')[-2].split('-')[-1])]

        # Select random time index
        if self.do_overfitting:
            time_idx = 0
        else:
            time_idx = randint(
                0, len(signal) - int(self.seg_length * self.dataset_fs))

        # Extract a subset of the signal
        signal = np.array(
            signal[time_idx: time_idx + int(self.seg_length * self.dataset_fs)],
            dtype=np.float
        )

        # Resample the frequency if necessary
        if self.fs != self.dataset_fs:
            signal = resample_frequency(signal, seg_len=self.seg_length,
                                        new_fs=self.fs, old_fs=self.dataset_fs)

        # Preprocess signal with a high pass filter
        if self.high_pass:
            signal = obspy_highpass(signal, df=self.fs)

        # Preprocess signal with a notch filter
        if self.notch:
            signal = notch_filter(signal, fs=self.fs)

        # Convert to tensor and reshape from (-1, 1) to (1, -1)
        signal = torch.tensor(signal, dtype=torch.float32).view(1, -1)

        # Instance normalization
        if self.instance_normalization:
            mean = signal.mean(dim=1)
            std = signal.std(dim=1)
            signal = (signal - mean) / (std + 1e-9)

        # Data augmentation, add noise
        if self.intensity_range is not None:
            signal *= np.random.uniform(*self.intensity_range)

        return signal, gender
