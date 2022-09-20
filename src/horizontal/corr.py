import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, basename
from glob import glob
import pickle
from scipy.stats import pearsonr
from scipy.signal import periodogram, filtfilt, correlate, correlation_lags
import seaborn as sns
from tqdm import tqdm
from itertools import combinations
from datetime import timedelta

from filtering import freq_filt

sps = 200
banner_coords = (44.3, -115.233)

result_dir = '/bsuscratch/zacharykeskinen/data/infrasound/psd_results'
data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'
hor_data = join(data_dir, 'ada-horizontal_2')

sps = 200

def norm_correlate(arr1, arr2):
    arr1 = freq_filt(arr1, 1, "highpass")
    arr2 = freq_filt(arr2, 1, "highpass")
    c = correlate((arr1 - np.mean(arr1))/np.std(arr1), (arr2 - np.mean(arr2))/np.std(arr2), 'full') / min(len(arr1), len(arr2))
    l = correlation_lags(arr1.size, arr2.size,)
    return c, l

with open(join(data_dir,'merged', 'horizontal', 'ada2.pkl'), 'rb') as f:
    a = pickle.load(f)

step = '1.005 second'
i = 0
for t1 in tqdm(np.arange(start = a.index.min(), stop = a.index.max() - pd.Timedelta(step), step = pd.Timedelta(step))):
    t1 = pd.Timestamp(t1).tz_localize('UTC')
    t2 = t1 + pd.Timedelta(step)
    sub = a.loc[(a.index < t2) & (a.index > t1)]
    for c1, c2 in combinations(sub.columns, 2):
        c, l = norm_correlate(sub[c1], sub[c2])
        if np.max(c) > 0.7:
            plt.plot(l, abs(c))
            name = f'{t1}_{c1}and{c2}'
            name = name.replace(' ','-')
            name = name.replace('.',':')
            #print(name)
            plt.title(name)
            plt.savefig(join('/bsuhome/zacharykeskinen/infrasound/figures/horizontal',name))
