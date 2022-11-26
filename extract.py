import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import struct

dsd = pd.read_excel('disdrometer.xlsx', sheet_name='BJXSY')
sites = (dsd.values)[:20, 0]
# xs = (dsd.values)[:, 5]
# ys = (dsd.values)[:, 6]
azis = np.round((dsd.values)[:20, 7].astype(np.float64)).astype(np.int64)
gates = (dsd.values)[:20, 9].astype(np.int64)-1
azis = azis[gates<1000]
gates = gates[gates<1000]
#%%

path = 'out'
ls = os.listdir(path)
d = []
for fname in ls:
    if 'qc' in fname:
        fpath = os.path.join(path, fname)
        
        with open(fpath, 'rb') as data_file:    
            values = np.array(struct.unpack('f'*(9*360*1000*5), data_file.read()))
        size = 9*360*1000
        zh = values[:size].reshape(9,360,1000)
        zdr = values[size:2*size].reshape(9,360,1000)
        phidp = values[2*size:3*size].reshape(9,360,1000)
        kdp = values[3*size:4*size].reshape(9,360,1000)
        cc = values[4*size:5*size].reshape(9,360,1000)
        
        d.append(zh[1, azis, gates])
        
#%%
# path = '20180716/FS/nc'
# ls = os.listdir(path)
# data = []
# labels = []
# for fname in ls:
#     if fname.endswith('nc'):
#         print(fname)
#         fpath = os.path.join(path, fname)

        
#         f = nc.Dataset(fpath, 'r')
#         grid_zh = np.array(f.variables['grid_zh'])
#         grid_zdr = np.array(f.variables['grid_zdr'])
#         grid_phidp = np.array(f.variables['grid_phidp'])
#         grid_kdp = np.array(f.variables['grid_kdp'])
#         grid_cc = np.array(f.variables['grid_cc'])
#         f.close()
        
#         ts = np.zeros_like(sites)
#         ts[:] = ''.join(fname.split('_')[1:])[:-3]
#         labels.append(np.transpose(np.array([sites, ts])))
        
        
#         i = np.int64(xs/75)
#         j = np.int64(ys/75)
#         d = np.array([grid_zh[0, i, j], grid_zdr[0, i, j], grid_phidp[0, i, j], grid_kdp[0, i, j], grid_cc[0, i, j],
#                       grid_zh[1, i, j], grid_zdr[1, i, j], grid_phidp[1, i, j], grid_kdp[1, i, j], grid_cc[1, i, j],
#                       grid_zh[2, i, j], grid_zdr[2, i, j], grid_phidp[2, i, j], grid_kdp[2, i, j], grid_cc[2, i, j],
#                       grid_zh[3, i, j], grid_zdr[3, i, j], grid_phidp[3, i, j], grid_kdp[3, i, j], grid_cc[3, i, j]])
#         data.append(np.transpose(d))

# data = np.array(data).reshape(-1, 20)
# labels = np.array(labels).reshape(-1,2)
            
# df = pd.DataFrame(np.hstack([labels, data]))
# df.to_excel('dsd-BJXFS-radar2.xlsx')
