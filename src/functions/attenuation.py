import os
from os.path import join, exists, dirname, basename
from glob import glob
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from filtering import freq_filt
from correlate import zero_lag_correlate
from conversions import dB_convert

def signal_power_calc(arr1, arr2, z, out, sps = 200, threshold = 0.8):
    idx = out>threshold
    arr1 = arr1[idx]
    l_time = arr1.size/sps
    arr1_p = np.sum((arr1**2)/sps)/(l_time)
    arr2 = arr2[idx]
    l_time = arr2.size/sps
    arr2_p = np.sum((arr2**2)/sps)/(l_time)

    return np.abs(arr1_p - arr2_p)

def attenuation_coefficent(z, a_0, a_z):
    """
    z: depth in meters
    a_0: amplitude at 0 meters
    a_z: amplitude at z meters
    """
    # return -np.log(a_z/a_0) / z
    return 

def calculate_correlated_attenuation_coef(arr1, arr2, z, out, threshold = 0.8):
    idx = out>threshold
    arr1 = np.abs(arr1[idx])
    arr2 = np.abs(arr2[idx])
    mean_attenuation = np.nanmean(attenuation_coefficent(z, arr1, arr2))
    return mean_attenuation

def snow_attenuation(h_dic, wind_s, hs, fc_low = 1, fc_high = 20, sps = 200, threshold = 0.8):
    res = {}
    arr_d = {}
    for h, f in h_dic.items():
        if h == 2:
            arr_fa = pd.read_parquet(f)['pa'].values
            arr_fa = freq_filt(arr_fa, (fc_low, fc_high), kind = 'bandpass')
        else:
            if h < hs:
                arr_d[h] = freq_filt(pd.read_parquet(f)['pa'].values, (fc_low, fc_high), kind = 'bandpass')
                if arr_d[h].size != 17280000:
                    return None
    out = zero_lag_correlate(arr_fa, arr_d[0.33], wind_s, sps = sps)
    out[:wind_s*sps] = 0
    out[-wind_s*sps:] = 0

    for (h1, arr1), (h2, arr2) in itertools.combinations(arr_d.items(), r = 2):
        ## This calculates using equation 1 in manuscript
        # attenuation_coef = calculate_correlated_attenuation_coef(arr1, arr2, abs(h1 - h2), out, threshold=threshold)
        # res[f'{h1}-{h2}'] = attenuation_coef

        ## Calculates the difference in power between the two sensors for coherent signals
        power_diff = signal_power_calc(arr1, arr2, abs(h1 - h2), out, sps = 200, threshold = 0.8)
        res[f'{h1}-{h2}'] = power_diff
    return res