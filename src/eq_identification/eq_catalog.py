import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from os.path import join
import pickle
from shapely.geometry import Point, box
from shapely.ops import transform
import pyproj
from correlate import zero_lag_correlate
from filtering import freq_filt
from tqdm import tqdm

data_dir = '/bsuscratch/zacharykeskinen/data/infrasound/array_data'
with open(join(data_dir, 'merged/all_days'), 'rb') as f:
    days = pickle.load(f)
sps = 200

banner_coords = (44.3, -115.233)
url = f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2021-12-01&endtime=2022-06-15&latitude={banner_coords[0]}&longitude={banner_coords[1]}&maxradiuskm=30'
df = pd.read_csv(url)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs = 'EPSG:4326')
res = gdf.loc[:, ['time', 'geometry', 'depth', 'mag', 'id']]

# Calculate Distance to Snotel from Source
wgs84 = pyproj.CRS('EPSG:4326')
itm = pyproj.CRS('EPSG:8826') # Idaho Tranvserse Mercator - https://epsg.io/8826
project = pyproj.Transformer.from_crs(wgs84, itm, always_xy=True).transform
wgs_snotel = Point(*banner_coords[::-1])
itm_snotel = transform(project, wgs_snotel)
res_geo = res.to_crs('EPSG:8826')
res.loc[:, 'distance_km'] = res_geo.distance(itm_snotel)/1000 # convert m to km
# Calculate Azimuth
geodesic = pyproj.Geod(ellps='WGS84')
for i,r in res.iterrows():
    fwd_azimuth,_,_ = geodesic.inv(wgs_snotel.x, wgs_snotel.y, r.geometry.x, r.geometry.y)
    res.loc[i, 'azimuth'] = fwd_azimuth

for i, r in tqdm(res.iterrows(), total = len(res)):
        dt = pd.to_datetime(r.time).strftime('%Y-%m-%d')
        if dt in days.keys():
            day = days[dt]
            if 0.33 in day.keys():
                s = pd.to_datetime(r.time)
                ps = s - pd.Timedelta('20 seconds') 
                e = s + pd.Timedelta('20 seconds')
                d = {}
                d['lower'] = pd.read_parquet(day[0.33])
                d['upper'] = pd.read_parquet(day[1])
                try:
                    pre = freq_filt(d['lower'].loc[ps:s, :].values.ravel(), fc = 1, kind = 'highpass', sps = sps)
                    pre_mean = np.nanmean(pre**2)
                    for k in d.keys():
                        d[k] = d[k].loc[s:e, :].values.ravel()
                        d[k] = freq_filt(d[k], fc = 1, kind = 'highpass', sps = sps)
                    corr = zero_lag_correlate(d['lower'], d['upper'], 2)[:-10]
                    max_corr = np.nanmax(corr)
                    res.loc[i, 'max_corr'] = max_corr
                    res.loc[i, 'mean_corr'] = np.nanmean(corr)
                    res.loc[i, 'corr_95'] = np.nanquantile(corr, 0.95)
                    eq_mean = np.nanmean(d['lower']**2)
                    mean_ratio = np.abs(eq_mean/pre_mean)
                    res.loc[i, 'mean_ratio'] = mean_ratio
                    if np.nanmax(corr) > 0.5 and mean_ratio > 1.5 and np.nanmean(corr) < 0.9:
                        res.loc[i, 'selected'] = 1
                    else:
                        res.loc[i, 'selected'] = 0
                except ValueError:
                    pass
res.loc[np.isnan(res.selected), 'selected'] = 0

res.to_csv('/bsuscratch/zacharykeskinen/data/infrasound/eq_catalog/selected.csv')