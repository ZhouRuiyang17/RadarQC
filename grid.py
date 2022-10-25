import datetime
_sttime = datetime.datetime.now()

from pyproj import Proj
from pyproj import CRS
import numpy as np
import netCDF4 as nc
#%%
# 北京大兴亦庄SA波段
lon_radar = 116.4719
lat_radar = 39.8088
h_radar = 0
e_radar = [0.5, 1.45, 2.4, 3.35, 4.3, 6.0, 9.9, 14.6, 19.5]
reso = 1

_deg2rad = np.pi/180
_rad2deg = 180/np.pi

#%%
xs = ys = np.arange(-230, 231, 1) # np.hstack([np.arange(-230, 0), np.arange(1,231)])
hs = np.arange(0.5, 4.5, 1)
grid = []
grid_sp = []
grid_x = []
grid_y = []
grid_h = []
grid_e = []
grid_a = []
grid_r = []
for i in range(4):
    for j in range(461):
        for k in range(461):
            h = hs[i]
            x = xs[j]
            y = ys[k]
            
            s = (x**2+y**2)**0.5
            r = (s**2+h**2)**0.5
            e = np.arcsin(h/r)*_rad2deg
            if s == 0:
                a = np.nan
            else:
                a = np.arcsin(abs(x)/s)*_rad2deg
                if x>=0:
                    if y>=0:
                        a = a
                    elif y<0:
                        a = 180-a
                elif x<0:
                    if y<0:
                        a = 180+a
                    elif y>=0:
                        a = 360-a
                
            grid.append([h, x, y])
            grid_sp.append([e, a, r])
            grid_x.append(x)
            grid_y.append(y)
            grid_h.append(h)
            grid_e.append(e)
            grid_a.append(a)
            grid_r.append(r)
grid_x = np.array(grid_x).reshape(4, 461, 461)
_contain_nan = (True in np.isnan(grid_x))
print(_contain_nan)
grid_y = np.array(grid_y).reshape(4, 461, 461)
_contain_nan = (True in np.isnan(grid_y))
print(_contain_nan)
grid_h = np.array(grid_h).reshape(4, 461, 461)
_contain_nan = (True in np.isnan(grid_h))
print(_contain_nan)
grid_e = np.array(grid_e).reshape(4, 461, 461)
_contain_nan = (True in np.isnan(grid_e))
print(_contain_nan)
grid_a = np.array(grid_a).reshape(4, 461, 461)
_contain_nan = (True in np.isnan(grid_a))
print(_contain_nan)
grid_r = np.array(grid_r).reshape(4, 461, 461)
_contain_nan = (True in np.isnan(grid_r))
print(_contain_nan)
#%%
fname = '20180716/SA/nc/Z_RADR_I_Z9010_20180716000600_O_DOR_SA_CAP.nc'
f = nc.Dataset(fname, 'r')
print(f)
zh = np.array(f.variables['zh'])
f.close()
data = []
for i in range(len(grid_sp)):
    e, a, r = grid_sp[i]
    if np.isnan(a):
        d = np.nan
    else:
        a = round(a) # 0.65->1, 341.3->341
        r = int(r) # 26.7->26, 30.1->30
        if a==360:
            a=0
        
        if r>=230: # out of sphere
            d = -33
        else: # in sphere
            if e<0 or e>20: # ouf of min or max
                d = -33
            elif e>=0 and e<0.5:
                d = zh[0, a, r]
            elif e<=20 and e>19.5:
                d = zh[-1, a, r]
            else:
                for j in range(8):
                    e1 = e_radar[j]
                    zh1 = zh[j, a, r]
                    e2 = e_radar[j+1]
                    zh2 = zh[j+1, a, r]
                    if e>=e1 and e<=e2:
                        if abs(e-e1)<=0.5: # in e1 scan
                            d = zh1
                            break
                        elif abs(e-e2)<=0.5: # in e2 scan
                            d = zh2
                            break
                        else: # out of scan, needing interpolation
                            w1 = (e-e1)/(e2-e1)
                            w2 = (e2-e)/(e2-e1)
                            d = (w1*zh1+w2*zh2)/(w1+w2)
                            break
    data.append(d)
data1 = np.array(data).reshape(4,461,461)

#%%
import wradlib as wl
import matplotlib.pyplot as plt


fname = '20180716/composite/000533.nc'
f = nc.Dataset(fname, 'r')
print(f)
dbz = f.variables['DBZ']
dbz = np.array(dbz)[0, :, 400-230:400+231, 400-230:400+231]
f.close()

for i in range(0,4):
    fig = plt.figure(figsize = [10, 10])
    _, pm = wl.vis.plot_ppi(zh[i, :, :230], r = np.arange(0, 230))  
    cb = plt.colorbar(pm)
    plt.show()
    fig = plt.figure(figsize = [10, 10])
    plt.pcolormesh(xs+230, ys, np.transpose(data1[i]))
    plt.colorbar()
    plt.show()
    fig = plt.figure(figsize = [10, 10])
    plt.pcolormesh(xs+230, ys, dbz[i])
    plt.colorbar()
    plt.show()
_edtime = datetime.datetime.now()
print(_edtime-_sttime)