from pyproj import Transformer
from pyproj import Proj
from pyproj import CRS, Geod
import pandas as pd
import numpy as np
df = pd.read_excel('disdrometer.xlsx', sheet_name = 'BJXSY')
reso = 75#m
Re = 8500#km

deg2rad = np.pi/180

lat, lon = df.iloc[20, [1,2]]
p = Proj(proj='utm', zone=int(lon//6+31), ellps='WGS84', preserve_units=False)
x, y = p(lon, lat)
geod_wgs84 = CRS("epsg:4326").get_geod()
for i in range(len(df)):
    lat1, lon1 = df.iloc[i,[1,2]]
    x1, y1 = p(lon1, lat1)
    dx, dy = x1-x, y1-y
    df.iloc[i,3:7] = x1, y1, dx, dy

    # az12_wgs, az21_wgs, dist_wgs = geod_wgs84.inv(boston_lon, boston_lat, portland_lon, portland_lat)
    theta, _, s = geod_wgs84.inv(lon, lat, lon1, lat1)
    if theta<0:
        theta = theta+360    
    df.iloc[i, 7], df.iloc[i, 8] = theta, s

    for j in range(1000):
        rt = int(s/reso+j)*reso/1000 #m->km
        Reh = (Re**2+rt**2-2*np.cos(deg2rad*(90+1.5))*Re*rt)**0.5
        alpha = np.arccos((Re**2+Reh**2-rt**2)/2/Re/Reh)
        st = alpha*Re*1000#km->m
        if (st-s) < reso and st>s:
            df.iloc[i, 9], df.iloc[i, 10], df.iloc[i, 11] = int(s/reso+j), rt*1000, st
            break
        
    
    
nd = df.values