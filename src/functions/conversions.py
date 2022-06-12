import numpy as np

def dB_convert(series):
    return 10*np.log10(series.values/np.nanmax(series.values))