import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, basename
import pickle
from scipy.stats import pearsonr
from scipy.signal import periodogram, filtfilt, spectrogram, welch
import seaborn as sns

from filtering import freq_filt

banner_coords = (44.3, -115.233)
res = pd.read_csv('/bsuscratch/zacharykeskinen/data/infrasound/eq_catalog/selected_v2.csv')
from shapely import wkt
res['geometry'] = res['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(res, geometry = 'geometry', crs = 'EPSG:4326')

result_dir = '/bsuscratch/zacharykeskinen/data/infrasound/psd_results/windows'
data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'
with open(join(data_dir, 'merged/all_days'), 'rb') as f:
    days = pickle.load(f)
sps = 200

############################### Welch #####################################
# Get welch earhtquake averages
window_start = [0, 3, 5, 10]
windows_end = [5,10,15,30]
for start in window_start:
    for end in windows_end:
        if start >= end:
            print('nope')
            break
        n = 0
        avg_Pxx = np.array([])
        for i, r in res[res.selected == 1].iterrows():
            if i == 0:
                dt = pd.to_datetime(r.time).strftime('%Y-%m-%d')
                day = days[dt]
                sig = {}
                s = pd.to_datetime(r.time) + pd.Timedelta(f'{start} second')
                e = s + pd.Timedelta(f'{end} second')
                try:
                    if 0.33 in day.keys() and 1.33 in day.keys() and day['snotel']['Snow Depth (cm) Start of Day Values'] > 133:
                        for name, fp in day.items():
                            if name != 'snotel':
                                arr = freq_filt(pd.read_parquet(fp)[s:e].values.ravel(), 1, kind = 'highpass')
                                arr = arr[:2000]
                                f, Pxx = welch(arr, sps, scaling = 'density', window = 'hann')
                                # Pxx = filtfilt([1,1,1,1,1],5, Pxx)
                                sig[name] = Pxx
                        df = pd.DataFrame(sig)
                        df.index = f
                        if avg_Pxx.size == 0:
                            avg_Pxx = df
                        else:
                            avg_Pxx = avg_Pxx + df
                            n +=1
                except ValueError as e:
                    print(dt)
                    print(e)

        avg_Pxx = avg_Pxx/n

        with open(join(result_dir, f'welch_{start}_{end}.pkl'), 'wb') as f:
            pickle.dump(avg_Pxx, f)