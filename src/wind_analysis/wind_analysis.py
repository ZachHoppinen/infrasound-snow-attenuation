import os
from os.path import join, exists, basename, dirname, expanduser
from glob import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from os.path import join, exists, basename, dirname
from scipy.signal import welch
import multiprocessing
from multiprocessing import Pool, cpu_count
from datetime import datetime

from filtering import freq_filt
from correlate import zero_lag_correlate

print('Loading dataframes')
canyon = pd.read_csv('/bsuscratch/zacharykeskinen/data/infrasound/snotel/canyon_wx.csv', comment = '#', parse_dates=['Date_Time'], index_col = ['Date_Time'])
units = canyon.iloc[0]
canyon = canyon.iloc[1:]
# convert all columns of DataFrame
canyon = canyon.apply(pd.to_numeric, errors = 'ignore')
canyon = canyon.loc[:pd.to_datetime('2022-05-15')]
canyon = canyon.tz_convert('UTC')

snotel_fp = '/bsuscratch/zacharykeskinen/data/infrasound/snotel/banner_snotel_results.csv'
snotel_fp = '/bsuscratch/zacharykeskinen/data/infrasound/snotel/banner_snotel_results.csv'
snotel = pd.read_csv(snotel_fp, comment='#', index_col=['Date'], parse_dates=['Date'])
for c in snotel.columns:
    snotel[c] = snotel[c].astype('f4')
snotel['Snow Water Equivalent'] = snotel['Snow Water Equivalent (mm) Start of Day Values']/1000
snotel['Snow Depth'] = snotel['Snow Depth (cm) Start of Day Values']/100
snotel['Average Air Temp'] = snotel['Air Temperature Average (degC)']

result_dir = '/bsuscratch/zacharykeskinen/data/infrasound/wind_results'
tmp_dir = join(result_dir, 'tmp')
data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'
with open(join(data_dir, 'merged/all_days'), 'rb') as f:
    days = pickle.load(f)
sps = 200

print('Starting analysis...')

def process(day):
    full = pd.DataFrame()
    d, fps = day
    winds = canyon.loc[pd.to_datetime(d + 'T00:00:00Z'):pd.to_datetime(d + 'T23:59:00Z'), 'wind_speed_set_1']
    d = pd.to_datetime(d)
    if d < pd.to_datetime('2022-04-30'):
        sd = snotel.loc[d, 'Snow Depth (cm) Start of Day Values']/100
        try:
            for s, wind_speed in winds.iteritems():
                e = s + pd.Timedelta('1 hour')
                for h, fp in fps.items():
                    if h != 'snotel':
                        res = {}
                        res['sd_delta'] = [sd - h]
                        res['wind'] = [wind_speed]
                        arr = freq_filt(pd.read_parquet(fp)[s:e].values.ravel(), 1, kind = 'highpass')
                        arr = arr[60*60*sps]
                        # power = np.sum(arr**2)/arr.size
                        power = welch(arr, fs = sps)
                        res['power'] = [power]
                        if 2 in fps.keys() and h !=  2:
                            res['cor'] = [np.nanmean(zero_lag_correlate(fps[2],fps[h], wind_s = 1))]
                        elif 0.33 in fps.keys() and h!= 0.33:
                            res['cor'] = [np.nanmean(zero_lag_correlate(fps[0.33],fps[h], wind_s = 1))]
                        elif 1.33 in fps.keys() and h != 1.33:
                            res['cor'] = [np.nanmean(zero_lag_correlate(fps[1.33],fps[h], wind_s = 1))]
                        elif 1 in fps.keys() and h != 1:
                            res['cor'] = [np.nanmean(zero_lag_correlate(fps[1],fps[h], wind_s = 1))]
                        else:
                            res['cor'] = [np.nan]
                        df = pd.DataFrame(res)
                        full = pd.concat([full, df])
        except ValueError as e:
            print(e)
        except IndexError as e:
            print(e)
    if full.size > 0:
        with open(join(tmp_dir, d.strftime('%Y-%m-%d')+'.pkl'), 'wb') as f:
            pickle.dump(full, f)


# loc_old = None
start_time = datetime.now()

os.makedirs(tmp_dir, exist_ok= True)

pool = Pool()                         # Create a multiprocessing Pool
pool.map(process, iter(days.items()))

res = pd.DataFrame()
for f in glob(join(tmp_dir, '*')):
    with open(f, 'rb') as f:
        df = pickle.load(f)
    res = pd.concat([res, df])

with open(join(result_dir, 'windv2.pkl'), 'wb') as f:
    pickle.dump(res, f)

end_time = datetime.now()
print(f'Run Time: {end_time - start_time}')
