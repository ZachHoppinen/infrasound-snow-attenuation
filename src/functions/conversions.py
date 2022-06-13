import numpy as np

def dB_convert(arr):
    return 10*np.log10(arr/np.nanmax(arr))