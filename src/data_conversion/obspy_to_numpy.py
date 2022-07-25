"""
Script converts all mseed files into numpy array. Each array is detrended by fitting a linear function and subtracting that line to remove the DC offset.
"""

import obspy

import numpy as np
import pandas as pd
from glob import glob
import os
from os.path import join, basename, exists
from fastparquet import write
from datetime import datetime

def main():
    ac_calib = 8.2928e-05
    name_dic = {'be4':'lower','a3m':'upper','ad8':'horizontal','ada':'horizontal_2'}
    local_data_dir = '/home/zacharykeskinen/Documents/infrasound/data/banner/infrasound/processed'
    target_dir = '/home/zacharykeskinen/Documents/infrasound/array_data'
    assert exists(target_dir)
    fps = glob(join(local_data_dir, '*'))
    fps = [f for f in fps if '.1.' not in f]
    array_names = np.unique([basename(f)[2:5] for f in fps])
    assert set(array_names) == set(name_dic.keys()), 'Missing an array in dict?'
    for ext, desc in name_dic.items():
        print(f'Starting {desc} array')
        array_dir = join(target_dir, f'{ext}-{desc}')
        os.makedirs(array_dir, exist_ok= True)
        array_fps = sorted([f for f in fps if ext in f])
        for i, fp in enumerate(array_fps):
            tr = obspy.read(fp)[0]
            stats = tr.stats
            channel = int(stats.channel.replace('p','')) + 1
            out_fp = join(array_dir, f'{stats.starttime.date}_c{channel}.parq')
            if not exists(out_fp):
                tr.detrend("linear")
                arr = np.array(tr.data * ac_calib)
                t = [datetime.fromtimestamp(t) for t in tr.times("timestamp")]
                res = pd.DataFrame(arr, index = t, columns = ['pa'])
                res.index = res.index.tz_localize('US/Mountain').tz_convert('UTC')

                ## Parquet
                write(out_fp, res, compression = 'GZIP')
            if i%10 == 0:
                print(f'Completed {i} of {len(array_fps)}')


if __name__ == '__main__':
    main()
