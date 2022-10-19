import pickle
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import numpy as np
import pandas as pd

data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'
with open(join(data_dir, 'merged/all_days'), 'rb') as f:
    days = pickle.load(f)
sps = 200

from filtering import freq_filt
final = np.zeros((24, 5))
i = 0
col_h = {0.33:0, 0.66:1, 1:2, 1.33:3, 2.0:4}
totals = dict(zip(range(24), np.zeros(24)))
for day, fps in days.items():
    day = pd.to_datetime(day)
    if fps['snotel']['Snow Depth (cm) Start of Day Values'] > 140:
        if 0.33 in fps.keys() and 2.0 in fps.keys():
            # if day.date() < pd.to_datetime('2022-01-23').date():
            for h, fp in fps.items():
                if h != 'snotel':
                    col = col_h[h]
                    for si in range(24):
                        s = day + pd.Timedelta(f'{si} hour')
                        s = s.tz_localize('UTC')
                        e = s + pd.Timedelta(f'1 hour')
                        series = pd.read_parquet(fp)[s:e].values.ravel()
                        if len(series) > 60*60*sps-10:
                            arr = freq_filt(series, 1, kind = 'highpass')
                            p = np.sum(arr**2)
                            final[si, col] += p
                            if h == 2.0:
                                totals[si] += 1

final2 = final/np.array(list(totals.values()))[:, None]

with open(join('/bsuhome/zacharykeskinen/infrasound/results/hourly_average', 'v1'), 'wb') as f:
    pickle.dump(final2, f)