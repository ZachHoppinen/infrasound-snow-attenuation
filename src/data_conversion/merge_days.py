import os
from os.path import join, exists, dirname, basename
from glob import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '/home/zacharykeskinen/Documents/infrasound/array_data'
data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'


arrays = glob(join(data_dir, '*'))
fps = {}
for array in arrays:
    if 'horizontal' not in array:
        fps[basename(array).split('-')[-1]] = sorted(glob(join(array, '*')))

height_dic = {'be4-lower_c1':0.33,'be4-lower_c2':0.66, 'be4-lower_c3':1, 'a3m-upper_c1':1.33, 'a3m-upper_c2':None, 'a3m-upper_c3':2}
def height_parse(f):
    ext = basename(f).split('_')[1].split('.')[0]
    parsed = f'{basename(dirname(f))}_{ext}'
    return height_dic[parsed]

upper_dates = np.unique([basename(d).split('_')[0] for d in fps['upper']])
lower_dates = np.unique([basename(d).split('_')[0] for d in fps['lower']])
dates = np.append(upper_dates, lower_dates)
res = {}
for date in dates:
    day_fps = {}
    for array, searched_fps in fps.items():
        arrays_fps = [f for f in searched_fps if date in f]
        height_fps = {}
        if arrays_fps:
            for f in arrays_fps:
                if height_parse(f):
                    height_fps[height_parse(f)] = f                    
            day_fps.update(height_fps)
    res[date] = day_fps

with open(join(data_dir, 'merged/all_days'), 'wb') as f:
    pickle.dump(res, f)

upper_dates = np.unique([basename(d).split('_')[0] for d in fps['upper']])
lower_dates = np.unique([basename(d).split('_')[0] for d in fps['lower']])
dates = np.intersect1d(lower_dates, upper_dates)
res = {}
for date in dates:
    day_fps = {}
    for array, searched_fps in fps.items():
        arrays_fps = [f for f in searched_fps if date in f]
        height_fps = {}
        if arrays_fps:
            for f in arrays_fps:
                if height_parse(f):
                    height_fps[height_parse(f)] = f                    
            day_fps.update(height_fps)
    res[date] = day_fps

with open(join(data_dir, 'merged/full_days'), 'wb') as f:
    pickle.dump(res, f)