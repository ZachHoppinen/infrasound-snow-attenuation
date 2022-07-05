"""
Converts dictionary of all days and full days to correct file paths for borah
"""

import os
from os.path import join, exists, dirname, basename
from glob import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Convert full day files

local_data_dir = '/home/zacharykeskinen/Documents/infrasound/array_data'
borah_data_dir = '/bsuhome/zacharykeskinen/scratch/data/infrasound/array_data'
with open(join(borah_data_dir, 'merged/full_days'), 'rb') as f:
    days = pickle.load(f)

for d, dic in days.items():
    for h, f in dic.items():
        days[d][h] = f.replace(local_data_dir, borah_data_dir)

with open(join(local_data_dir, 'merged/full_days'), 'wb') as f:
    pickle.dump(days, f)

## Convert all days files
with open(join(borah_data_dir, 'merged/all_days'), 'rb') as f:
    days = pickle.load(f)

for d, dic in days.items():
    for h, f in dic.items():
        days[d][h] = f.replace(local_data_dir, borah_data_dir)

with open(join(borah_data_dir, 'merged/all_days'), 'wb') as f:
    pickle.dump(days, f)