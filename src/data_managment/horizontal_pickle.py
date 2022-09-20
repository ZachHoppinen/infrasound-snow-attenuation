import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, basename
from glob import glob
import pickle
from scipy.stats import pearsonr
from scipy.signal import periodogram, filtfilt
import seaborn as sns
from tqdm import tqdm
from itertools import combinations

from filtering import freq_filt

sps = 200
banner_coords = (44.3, -115.233)

result_dir = '/bsuscratch/zacharykeskinen/data/infrasound/psd_results'
data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'
hor_data = join(data_dir, 'ada-horizontal_2')

sps = 200

res = []
oldname = ''
for fp in sorted(glob(join(hor_data, '*'))):
    name = basename(fp).strip('.parq')
    sen = name.split('_')[-1]
    day = name.split('_')[0]
    if day != oldname:
        oldname = day
        day_df = pd.DataFrame()
    day_df.loc[:, sen] = pd.read_parquet(fp)
    if sen == 'c3':
        res.append(day_df)

a = pd.concat(res)
print(a)

with open(join(data_dir,'merged', 'horizontal', 'ada2.pkl'), 'wb') as f:
    pickle.dump(a, f)