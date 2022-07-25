from os.path import join, exists, basename, dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from obspy import read
from scipy.signal import correlate, correlation_lags, butter, filtfilt, spectrogram
from numpy.fft import fft, fftfreq, ifft, fftshift

def prep_wx(wx_dir = '../../data/banner/wx/'):
    wx_fs = glob(join(wx_dir, '*STAND_MONTH*'))
    df = pd.DataFrame()
    for fp in wx_fs:
        df = df.append(pd.read_csv(fp, skiprows=3), ignore_index= True)
    df.Date = pd.to_datetime(df.Date)
    df = df.set_index(df.Date)
    df.loc[:, 'SWE_m'] = df['WTEQ.I-1 (in) ']*0.0254
    df.loc[:, 'SD_m'] = df['SNWD.I-1 (in) ']*0.0254
    df = df.drop_duplicates()

    return df

def mseed2arr(fp, filtered = False, ac_calib = 8.2928e-05):

    tr = read(fp)[0]
    arr = tr.data
    arr = arr * ac_calib
    if filtered:
        arr = freq_filt(arr, 2, 1, 'highpass', sps = tr.stats['sampling_rate'])
    return arr

def prep(in_dir, ext = '*'):
    name_dic = {'be4':'lower','a3m':'upper','ad8':'failed array'}
    height_dic = {'lower-p0':0.33,'lower-p1':0.66,'lower-p2':1,'upper-p0':1.33,'upper-p1':np.nan,'upper-p2':2}
    assert exists(in_dir)
    l = glob(join(in_dir, ext))
    r = []
    for i in l:
        i = basename(i)
        j = i[5:11]
        if j not in r:
            r.append(j)
    r.sort()

    days = {}
    day_stats = {}
    for day in r:
        ls = glob(join(in_dir, '*'+day+'*'))
        res = {}
        for fp in ls:
                name = name_dic[basename(fp).replace(day,'')[2:5]]
                if name != 'failed array':
                    tr = read(fp)[0]
                    stats = tr.stats
                    channel = stats['channel']
                    name_channel = f'{name}-{channel}'
                    height = height_dic[name_channel]
                    res[height] = fp
                    # if not np.isnan(height):
                    #     arr = tr.data
                    #     arr = arr * ac_calib
                    #     arr = arr - np.nanmean(arr)
                        # res[height] = arr
        day_stats[day] = stats
        days[day] = res

    return days, day_stats