import os
from os.path import join, basename, expanduser
from glob import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count

from attenuation import snow_attenuation

print('Starting snow layer attenuation analysis...')
print(f'Creating {cpu_count()}-process pool')

local_data_dir = '/bsuhome/zacharykeskinen/scratch/data/infrasound/array_data'
with open(join(local_data_dir, 'merged/full_days'), 'rb') as f:
    days = pickle.load(f)
days = iter(days.values())

result_dir = expanduser(f'~/infrasound/results/attenuation/')
tmp_dir = join(result_dir, 'tmp')
os.makedirs(tmp_dir, exist_ok= True)

def process(fps):
    res = snow_attenuation(fps, 10, hs = 1.4, threshold = 0.7, fc_low=1, fc_high=20)
    day = basename(fps[0.33]).replace('_c1.parq','')
    with open(join(tmp_dir, day), 'wb') as f:
        pickle.dump(res, f)

start_time = datetime.now()
print(f'Starting at {start_time}')
print('Running pooled process')

pool = Pool()                         # Create a multiprocessing Pool
pool.map(process, days)

print('Combining tmp dataframes.')
res = pd.DataFrame()
for fp in glob(join(tmp_dir, '*')):
    with open(fp, 'rb') as f:
        dic = pickle.load(f)
    if dic:
        res = pd.concat([res, pd.DataFrame.from_records([dic], index = [basename(fp)])])

with open(join(result_dir, 'attenuation_1-20_power.pkl'), 'wb') as f:
    pickle.dump(res, f)

end_time = datetime.now()
print(f'Run Time: {end_time - start_time}')


