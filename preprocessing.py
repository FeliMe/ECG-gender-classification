import numpy as np

from obspy.signal.filter import highpass
from scipy.signal import resample, lfilter, iirnotch


def resample_frequency(signal, seg_len, new_fs, old_fs):
    """
    Resample the signal to a new frequency and keeping the time fixed

    args:
        signal (np.array): ecg signal
        seg_len (int): length of the signal in seconds
        new_fs (int): frequency after resampling
        old_fs (int): frequency before resampling

    returns:
        resampled signal (np.array)
    """
    if seg_len is None:
        num = int(len(signal) / old_fs * new_fs)
    else:
        num = int(seg_len * new_fs)
    return resample(signal, num=num)


def obspy_highpass(signal, df, freq=3.6, corners=4):
    """
    Butterworth-Highpass Filter, removing data below certain frequency

    args:
        signal (np.array): ecg signal
        df (int): sampling frequency of signal
        freq (int): cut-off frequency
        corners (int): number of corners used for filtering

    returns:
        filtered signal (np.array)
    """
    # Prevent high pass artifacts by moving the signal to start at around 0
    signal -= signal[:50].mean()
    ret = np.zeros_like(signal)
    for i in range(signal.shape[-1]):
        ret[:, i] = highpass(
            data=signal[:, i],
            freq=freq,
            df=df,
            corners=corners
        )
    return ret


def notch_filter(signal, fs, filter_range=[50, 60]):
    """
    Notch filter (band stop filter), removing signal of a certain frequency

    args:
        signal (np.aaray): ecg signal
        fs (int): sampling frequency of the signal
        filter_range (list): min and max frequency to be filtered

    returns:
        filtered signal (np.array)
    """
    # notch filter
    for f0 in filter_range:  # Frequency to be removed from signal (Hz)
        Q = 30.0  # Quality factor
        w0 = f0 / (fs / 2)  # Normalized Frequency
        b, a = iirnotch(w0, Q)
        for i in range(signal.shape[-1]):
            signal[:, i] = lfilter(b, a, signal[:, i])

    return signal
