from pyproj import Transformer
from pyproj import Proj
from pyproj import CRS, Geod
import pandas as pd
df = pd.read_excel('info_disdrometer - utm.xlsx', sheet_name = 'BJXSY')


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
    theta, _, r = geod_wgs84.inv(lon, lat, lon1, lat1)
    if theta<0:
        theta = theta+360
    df.iloc[i, -2], df.iloc[i, -1] = theta, r

nd = df.values