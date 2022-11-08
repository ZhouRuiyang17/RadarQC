import datetime
_sttime = datetime.datetime.now()

from pyproj import Proj
from pyproj import CRS
import numpy as np
import netCDF4 as nc
import wradlib as wl
import matplotlib.pyplot as plt

e_radar = [0.5, 1.45, 2.4, 3.35, 4.3, 6.0, 9.9, 14.6, 19.5]
_deg2rad = np.pi/180
_rad2deg = 180/np.pi

# # 北京大兴亦庄SA波段
# lon_radar = 116.4719
# lat_radar = 39.8088
# dia = 230
# reso = 1

# # BJXSY
# lon_radar = 116.61528
# lat_radar = 40.126945
# dia = 75
# reso = 0.075

#BJXFS
lon_radar = 116.19055
lat_radar = 39.690277
dia = 75
reso = 0.075

h_radar = 42


def generate_grid(diameter:float, resolution:float):
    dia = diameter
    reso = resolution
    num_rad = np.int32(dia/reso)
    
    grid_x = np.zeros(shape=(2*num_rad+1, 2*num_rad+1))
    grid_x[:] = np.arange(-dia, dia+reso, reso)
    
    grid_y = np.transpose(grid_x)
    
    grid_s = (grid_x**2+grid_y**2)**0.5
    
    grid_a = np.arcsin(abs(grid_x)/grid_s)*_rad2deg
    loc = np.where((grid_x>=0) & (grid_y<0))
    grid_a[loc] = 180 - grid_a[loc]
    loc = np.where((grid_x<0) & (grid_y<0))
    grid_a[loc] = 180 + grid_a[loc]
    loc = np.where((grid_x<0) & (grid_y>=0))
    grid_a[loc] = 360 - grid_a[loc]
    grid_a = np.array([grid_a, grid_a, grid_a, grid_a])
    
    grid_h = np.zeros_like(grid_a)
    grid_h[0] = 0.5
    grid_h[1] = 1.5
    grid_h[2] = 2.5
    grid_h[3] = 3.5
    
    grid_r = (grid_s**2+grid_h**2)**0.5
    
    grid_e = np.arcsin(grid_h/grid_r)*_rad2deg
    
    # _contain_nan = (True in np.isnan(grid_x))
    # print(_contain_nan)
    # _contain_nan = (True in np.isnan(grid_y))
    # print(_contain_nan)
    # _contain_nan = (True in np.isnan(grid_h))
    # print(_contain_nan)
    # _contain_nan = (True in np.isnan(grid_s))
    # print(_contain_nan)
    # _contain_nan = (True in np.isnan(grid_e))
    # print(_contain_nan)
    # _contain_nan = (True in np.isnan(grid_a))
    # print(_contain_nan)
    # _contain_nan = (True in np.isnan(grid_r))
    # print(_contain_nan)
    
    return grid_e, grid_a, grid_r


'''
dia = 75
reso = 0.075
num_rad = np.int32(dia/reso)

grid = []
grid_sp = []
grid_x = np.zeros(shape=(2*num_rad+1, 2*num_rad+1))
grid_x[:] = np.arange(-dia, dia+reso, reso)

grid_y = np.transpose(grid_x)

grid_s = (grid_x**2+grid_y**2)**0.5

grid_a = np.arcsin(abs(grid_x)/grid_s)*_rad2deg
loc = np.where((grid_x>=0) & (grid_y<0))
grid_a[loc] = 180 - grid_a[loc]
loc = np.where((grid_x<0) & (grid_y<0))
grid_a[loc] = 180 + grid_a[loc]
loc = np.where((grid_x<0) & (grid_y>=0))
grid_a[loc] = 360 - grid_a[loc]
grid_a = np.array([grid_a, grid_a, grid_a, grid_a])

grid_h = np.zeros_like(grid_a)
grid_h[0] = 0.5
grid_h[1] = 1.5
grid_h[2] = 2.5
grid_h[3] = 3.5

grid_r = (grid_s**2+grid_h**2)**0.5

grid_e = np.arcsin(grid_h/grid_r)*_rad2deg

_contain_nan = (True in np.isnan(grid_x))
print(_contain_nan)
_contain_nan = (True in np.isnan(grid_y))
print(_contain_nan)
_contain_nan = (True in np.isnan(grid_h))
print(_contain_nan)
_contain_nan = (True in np.isnan(grid_s))
print(_contain_nan)
_contain_nan = (True in np.isnan(grid_e))
print(_contain_nan)
_contain_nan = (True in np.isnan(grid_a))
print(_contain_nan)
_contain_nan = (True in np.isnan(grid_r))
print(_contain_nan)

'''
# #%%
# for i in range(4):
#     for j in range(2001):
#         for k in range(2001):
#             h = hs[i]
#             x = xs[j]
#             y = ys[k]
            
#             s = (x**2+y**2)**0.5
#             r = (s**2+h**2)**0.5
#             e = np.arcsin(h/r)*_rad2deg
#             if s == 0:
#                 a = np.nan
#             else:
#                 a = np.arcsin(abs(x)/s)*_rad2deg
#                 if x>=0:
#                     if y>=0:
#                         a = a
#                     elif y<0:
#                         a = 180-a
#                 elif x<0:
#                     if y<0:
#                         a = 180+a
#                     elif y>=0:
#                         a = 360-a
                
#             grid.append([h, x, y])
#             grid_sp.append([e, a, r])
#             grid_x.append(x)
#             grid_y.append(y)
#             grid_h.append(h)
#             grid_s.append(s)
#             grid_e.append(e)
#             grid_a.append(a)
#             grid_r.append(r)



def fill_grid(zh, zdr, phidp, kdp, cc, grid_e, grid_a, grid_r, diameter, resolution):
    dia = diameter
    reso = resolution
    num_rad = np.int32(dia/reso)
    
    grid_zh = []
    grid_zdr = []
    grid_phidp = []
    grid_kdp = []
    grid_cc = []
    for i in range(4):
        for j in range(2*num_rad+1):
            for k in range(2*num_rad+1):
                e, a, r = grid_e[i, j, k], grid_a[i, j, k], grid_r[i, j, k]

                if np.isnan(a):
                    d1 = d2 = d3 = d4 = d5 = np.nan
                else:
                    a = round(a) # 0.65->1, 341.3->341
                    gate = int(r/reso) # 26.7->26, 30.1->30
                    if a==360:
                        a=0
                    
                    if r>=dia: # out of sphere
                        d1 = -33
                        d2 = -8.125
                        d3 = -2
                        d4 = -5
                        d5 = -0.025
                    else: # in sphere
                        if e<0 or e>20: # ouf of min or max
                            d1 = -33
                            d2 = -8.125
                            d3 = -2
                            d4 = -5
                            d5 = -0.025
                        elif e>=0 and e<0.5:
                            d1 = zh[0, a, gate]
                            d2 = zdr[0, a, gate]
                            d3 = phidp[0, a, gate]
                            d4 = kdp[0, a, gate]
                            d5 = cc[0, a, gate]
                        elif e<=20 and e>19.5:
                            d1 = zh[-1, a, gate]
                            d2 = zdr[-1, a, gate]
                            d3 = phidp[-1, a, gate]
                            d4 = kdp[-1, a, gate]
                            d5 = cc[-1, a, gate]
                        else:
                            for ie in range(8):
                                e1 = e_radar[ie]
                                d11 = zh[ie, a, gate]
                                d21 = zdr[ie, a, gate]
                                d31 = phidp[ie, a, gate]
                                d41 = kdp[ie, a, gate]
                                d51 = cc[ie, a, gate]
                                
                                e2 = e_radar[ie+1]
                                d12 = zh[ie+1, a, gate]
                                d22 = zdr[ie+1, a, gate]
                                d32 = phidp[ie+1, a, gate]
                                d42 = kdp[ie+1, a, gate]
                                d52 = cc[ie+1, a, gate]
                                
                                if e>=e1 and e<=e2:
                                    if abs(e-e1)<=0.5: # in e1 scan
                                        d1 = d11
                                        d2 = d21
                                        d3 = d31
                                        d4 = d41
                                        d5 = d51
                                        break
                                    elif abs(e-e2)<=0.5: # in e2 scan
                                        d1 = d12
                                        d2 = d22
                                        d3 = d32
                                        d4 = d42
                                        d5 = d52
                                        break
                                    else: # out of scan, needing interpolation
                                        w1 = (e-e1)/(e2-e1)
                                        w2 = (e2-e)/(e2-e1)
                                        d1 = (w1*d11+w2*d12)/(w1+w2)
                                        d2 = (w1*d21+w2*d22)/(w1+w2)
                                        d3 = (w1*d31+w2*d32)/(w1+w2)
                                        d4 = (w1*d41+w2*d42)/(w1+w2)
                                        d5 = (w1*d51+w2*d52)/(w1+w2)
                                        break
                grid_zh.append(d1)
                grid_zdr.append(d2)
                grid_phidp.append(d3)
                grid_kdp.append(d4)
                grid_cc.append(d5)
    grid_zh = np.array(grid_zh).reshape(grid_r.shape)
    grid_zdr = np.array(grid_zdr).reshape(grid_r.shape)
    grid_phidp = np.array(grid_phidp).reshape(grid_r.shape)
    grid_kdp = np.array(grid_kdp).reshape(grid_r.shape)
    grid_cc = np.array(grid_cc).reshape(grid_r.shape)
    
    return grid_zh, grid_zdr, grid_phidp, grid_kdp, grid_cc


'''
data = []
for i in range(4):
    for j in range(2*num_rad+1):
        for k in range(2*num_rad+1):
            e, a, r = grid_e[i, j, k], grid_a[i, j, k], grid_r[i, j, k]

            if np.isnan(a):
                d = np.nan
            else:
                a = round(a) # 0.65->1, 341.3->341
                gate = int(r/reso) # 26.7->26, 30.1->30
                if a==360:
                    a=0
                
                if r>=dia: # out of sphere
                    d = -33
                else: # in sphere
                    if e<0 or e>20: # ouf of min or max
                        d = -33
                    elif e>=0 and e<0.5:
                        d = zh[0, a, gate]
                    elif e<=20 and e>19.5:
                        d = zh[-1, a, gate]
                    else:
                        for ie in range(8):
                            e1 = e_radar[ie]
                            zh1 = zh[ie, a, gate]
                            e2 = e_radar[ie+1]
                            zh2 = zh[ie+1, a, gate]
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
data1 = np.array(data).reshape(grid_r.shape)
'''

def grid(path, fname):
    
    fpath = os.path.join(path, fname)
    
    f = nc.Dataset(fpath, 'a')
    print(f)
    
    # zh = np.array(f.variables['zh'])
    zh_qc = np.array(f.variables['zh_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
    zdr_qc = np.array(f.variables['zdr_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
    phidp_qc = np.array(f.variables['phidp_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
    kdp_qc = np.array(f.variables['kdp_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
    cc_qc = np.array(f.variables['cc_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
    # f.close()
    
    grid_e, grid_a, grid_r = generate_grid(dia, reso)
    grid_zh, grid_zdr, grid_phidp, grid_kdp, grid_cc = fill_grid(zh_qc, zdr_qc, phidp_qc, kdp_qc, cc_qc, 
                                                                 grid_e, grid_a, grid_r, dia, reso)
    
    f.createDimension('x', int(2*dia/reso+1))
    f.createDimension('y', int(2*dia/reso+1))
    f.createDimension('z', 4)
    f.createVariable('x', np.float64, ('x'))
    f.variables['x'][:] = np.arange(-(dia), (dia+reso), reso)
    f.createVariable('y', np.float64, ('y'))
    f.variables['y'][:] = np.arange(-(dia), (dia+reso), reso)
    f.createVariable('z', np.float64, ('z'))
    f.variables['z'][:] = np.array([0.5, 1.5, 2.5, 3.5])

    f.createVariable('grid_zh', np.float64, ('z', 'y', 'x'))
    f.variables['grid_zh'][:] = grid_zh
    f.createVariable('grid_zdr', np.float64, ('z', 'y', 'x'))
    f.variables['grid_zdr'][:] = grid_zdr
    f.createVariable('grid_phidp', np.float64, ('z', 'y', 'x'))
    f.variables['grid_phidp'][:] = grid_phidp
    f.createVariable('grid_kdp', np.float64, ('z', 'y', 'x'))
    f.variables['grid_kdp'][:] = grid_kdp
    f.createVariable('grid_cc', np.float64, ('z', 'y', 'x'))
    f.variables['grid_cc'][:] = grid_cc
    
    f.close()

if __name__ == '__main__':
    # fname = '20180716/SA/nc/Z_RADR_I_Z9010_20180716035400_O_DOR_SA_CAP.nc'
    # fname = '20180716/SY/nc/BJXSY_20180716_000000.nc'
    path = '20180716/FS/nc'
    import os
    ls = os.listdir(path)
    for f in ls[:]:
        if f.endswith('.nc'):
            grid(path, f)
        
        # fname = path+f
        # f = nc.Dataset(fname, 'a')
        # print(f)
        # # zh = np.array(f.variables['zh'])
        # zh_qc = np.array(f.variables['zh_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
        # zdr_qc = np.array(f.variables['zdr_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
        # phidp_qc = np.array(f.variables['phidp_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
        # kdp_qc = np.array(f.variables['kdp_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
        # cc_qc = np.array(f.variables['cc_qc'])[[0, 2, 4, 5, 6, 7, 8, 9, 10]]
        # # f.close()
        
        # grid_e, grid_a, grid_r = generate_grid(dia, reso)
        # grid_zh, grid_zdr, grid_phidp, grid_kdp, grid_cc = fill_grid(zh_qc, zdr_qc, phidp_qc, kdp_qc, cc_qc, 
        #                                                              grid_e, grid_a, grid_r, dia, reso)
        
        # f.createDimension('x', int(2*dia/reso+1))
        # f.createDimension('y', int(2*dia/reso+1))
        # f.createDimension('z', 4)
        # f.createVariable('x', np.float64, ('x'))
        # f.variables['x'][:] = np.arange(-(dia), (dia+reso), reso)
        # f.createVariable('y', np.float64, ('y'))
        # f.variables['y'][:] = np.arange(-(dia), (dia+reso), reso)
        # f.createVariable('z', np.float64, ('z'))
        # f.variables['z'][:] = np.array([0.5, 1.5, 2.5, 3.5])
    
        # f.createVariable('grid_zh', np.float64, ('z', 'y', 'x'))
        # f.variables['grid_zh'][:] = grid_zh
        # f.createVariable('grid_zdr', np.float64, ('z', 'y', 'x'))
        # f.variables['grid_zdr'][:] = grid_zdr
        # f.createVariable('grid_phidp', np.float64, ('z', 'y', 'x'))
        # f.variables['grid_phidp'][:] = grid_phidp
        # f.createVariable('grid_kdp', np.float64, ('z', 'y', 'x'))
        # f.variables['grid_kdp'][:] = grid_kdp
        # f.createVariable('grid_cc', np.float64, ('z', 'y', 'x'))
        # f.variables['grid_cc'][:] = grid_cc
        
        # f.close()
        
    
    # fname = '20180716/composite/035330.nc'
    # f = nc.Dataset(fname, 'r')
    # print(f)
    # dbz = f.variables['DBZ']
    # dbz = np.array(dbz)[0, :, 400-230:400+231, 400-230:400+231]
    # f.close()
    
    # for i in range(4):
        
    #     fig = plt.figure(figsize = [10, 10])
    #     # _, pm = wl.vis.plot_ppi(zh[i, :, :230], r = np.arange(0, 230))  
    #     _, pm = wl.vis.plot_ppi(cc_qc[i], r = np.arange(0, 75, 0.075)) 
    #     cb = plt.colorbar(pm)
    #     plt.show()
        
        # fig = plt.figure(figsize = [10, 10])
        # x = y = np.arange(-75, 75.075, 0.075)
        # plt.pcolormesh(x, y+75, grid[i])
        # # x = y = np.arange(-230, 231, 1)
        # # plt.pcolormesh(x, y+230, grid[i])
        # plt.colorbar()
        # plt.show()
        
        # fig = plt.figure(figsize = [10, 10])
        # plt.pcolormesh(x, y, dbz[i])
        # plt.colorbar()
        # plt.show()
    
    
    
_edtime = datetime.datetime.now()
print(_edtime-_sttime)
