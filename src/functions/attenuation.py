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

def attenuation_coefficent(z, a_0, a_z):
    """
    z: depth in meters
    a_0: amplitude at 0 meters
    a_z: amplitude at z meters
    """
    return -np.log(a_z/a_0) / z

def calculate_correlated_attenuation_coef(arr1, arr2, z, out, threshold = 0.8):
    idx = out>threshold
    arr1 = np.abs(arr1[idx])
    arr2 = np.abs(arr2[idx])
    mean_attenuation = np.nanmean(attenuation_coefficent(z, arr1, arr2))
    return mean_attenuation

def snow_attenuation(h_dic, wind_s, hs, fc_low = 1, fc_high = 20, sps = 200):
    e = False
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
        attenuation_coef = calculate_correlated_attenuation_coef(arr1, arr2, abs(h1 - h2), out)
        res[f'{h1}-{h2}'] = attenuation_coef
    
    return res