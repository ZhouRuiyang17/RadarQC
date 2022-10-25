# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:37:18 2022

@author: admin
"""

from pyproj import Proj
from pyproj import CRS
import numpy as np
lon_radar = 116.61528
lat_radar = 40.126945
h_radar = 29 # 雷达天线海拔29m
e_radar = [0.5, 0.5, 1.45, 1.45, 2.4, 3.35, 4.3, 6.0, 9.9, 14.6, 19.5]
reso = 0.075 # 空间分辨率75m

# =============================================================================
# 建立坐标转换器
# =============================================================================
p = Proj(proj = 'utm', zone = int(lon_radar//6+31), ellps = 'WGS84', preserve_units=False)
x_radar, y_radar = p(lon_radar, lat_radar)
print(lon_radar, lat_radar)
print(x_radar, y_radar)
print(p(x_radar, y_radar, inverse = True))

# =============================================================================
# 建立网格
# =============================================================================
xs = np.zeros(shape = (461, 461)) # 格子数
xs[:] = np.arange(-230, 231)*1 # 格子距离
ys = np.transpose(xs)
hs = np.arange(0.5, 4.5, 1) # 天线以上的高度
data = np.zeros(shape = (4, 461, 461))

# =============================================================================
# 坐标转换
# =============================================================================
xs = xs + x_radar
ys = ys + y_radar
lons = np.zeros(shape = (461, 461))
lats = np.zeros(shape = (461, 461))
for i in range(461):
    for j in range(461):
        x = xs[i,j]
        y = ys[i,j]
        lons[i,j] = p(x, y, inverse = True)[0]
        lats[i,j] = p(x, y, inverse = True)[1]

azis = np.zeros(shape = (461, 461))
geod_wgs84 = CRS("epsg:4326").get_geod()
for i in range(461):
    for j in range(461):
        lon = lons[i,j]
        lat = lats[i,j]
        azis[i,j] = geod_wgs84.inv(lon_radar, lat_radar, lon, lat)[0]+360