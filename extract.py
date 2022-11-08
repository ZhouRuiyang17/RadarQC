import os
import netCDF4 as nc
import numpy as np
import pandas as pd

dsd = pd.read_excel('dsd-BJXFS.xlsx')
sites = (dsd.values)[:, 0]
xs = (dsd.values)[:, -2]*1000 # km->m
ys = (dsd.values)[:, -1]*1000

#%%

path = '20180716/FS/nc'
ls = os.listdir(path)
data = []
labels = []
for fname in ls:
    if fname.endswith('nc'):
        print(fname)
        fpath = os.path.join(path, fname)

        
        f = nc.Dataset(fpath, 'r')
        grid_zh = np.array(f.variables['grid_zh'])
        grid_zdr = np.array(f.variables['grid_zdr'])
        grid_phidp = np.array(f.variables['grid_phidp'])
        grid_kdp = np.array(f.variables['grid_kdp'])
        grid_cc = np.array(f.variables['grid_cc'])
        f.close()
        
        ts = np.zeros_like(sites)
        ts[:] = ''.join(fname.split('_')[1:])[:-3]
        labels.append(np.transpose(np.array([sites, ts])))
        
        
        i = np.int64(xs/75)
        j = np.int64(ys/75)
        d = np.array([grid_zh[0, i, j], grid_zdr[0, i, j], grid_phidp[0, i, j], grid_kdp[0, i, j], grid_cc[0, i, j],
                      grid_zh[1, i, j], grid_zdr[1, i, j], grid_phidp[1, i, j], grid_kdp[1, i, j], grid_cc[1, i, j],
                      grid_zh[2, i, j], grid_zdr[2, i, j], grid_phidp[2, i, j], grid_kdp[2, i, j], grid_cc[2, i, j],
                      grid_zh[3, i, j], grid_zdr[3, i, j], grid_phidp[3, i, j], grid_kdp[3, i, j], grid_cc[3, i, j]])
        data.append(np.transpose(d))

data = np.array(data).reshape(-1, 20)
labels = np.array(labels).reshape(-1,2)
            
df = pd.DataFrame(np.hstack([labels, data]))
df.to_excel('dsd-BJXFS-radar2.xlsx')
