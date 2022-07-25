from os.path import join, exists, basename, dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from numpy.fft import fft, fftfreq, ifft, fftshift
from scipy.signal import butter, filtfilt

def freq_filt(arr, fc, kind = 'lowpass', order = 2, sps = 200):
    """
    arr: numpy array to filter
    order: order of filter to use
    fc: corner frequency
    kind: type of filter. options include 'lowpass', 'highpass', 'bandpass', 'bandstop'
    """
    b, a = butter(order, fc, kind, fs = sps)
    return filtfilt(b, a, arr)

def high_pass_filter(arr, order, fc, sps = 200, kind = 'high'):
    b, a = butter(order, fc, kind, fs = sps)

    return filtfilt(b, a, arr)

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh