import numpy as np
import wradlib as wl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import netCDF4 as nc
import struct

zh_nodata = -33
zdr_nodata = -8.125
phidp_nodata = -2
kdp_nodata = -5
cc_nodata = -0.025


def ppi(data, prv, r=None):
    color = np.array([[0, 0, 255], [30, 144, 255], [0, 255, 255], 
                      [0, 255, 0], [0, 205, 0], [0, 139, 0], 
                      [255, 255, 0], [205, 173, 0], [255, 165, 0],
                      [255, 0, 0], [205, 0, 0], [139, 0, 0], 
                      [255, 20, 147], [85, 26, 139], [171, 130, 255]])/255
    if prv == 'zh':
        data[data<0] = np.nan # 为了好看
        level = np.arange(0, 80, 5)
        cb_label = '$Z_{H}$ (dBZ)'
    elif prv == 'zdr':
        data[data == zdr_nodata] = np.nan
        level = np.arange(-7, 9, 1)
        cb_label = '$Z_{DR}$'
    elif prv == 'phidp':
        level = np.arange(0, 160, int(160/16))
        cb_label = '$\phi_{DP}$ ($\degree$)'
    elif prv == 'kdp':
        data[data == kdp_nodata] = np.nan
        level = np.arange(0, 8, 0.5)
        cb_label = r'$K_{DP}$ ($\degree/km$)'
    elif prv == 'cc':
        data[data<0] = np.nan # 不会<0
        data[data>1] = np.nan # 不会>1
        color = color[:10]
        level = np.arange(0, 1.1, 0.1)
        cb_label = r'$\rho_{hv}$'
    cmap, norm = colors.from_levels_and_colors(level, color)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax, pm = wl.vis.plot_ppi(data, r, ax = ax, cmap = cmap, norm = norm)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')
    cb = fig.colorbar(pm, ax=ax, extend = 'neither')
    cb.set_label(cb_label)
    
    plt.show()



if __name__ == "__main__":
    # fnc = nc.Dataset('20180716/BJXFS_20180716_000000.nc', 'r')
    # zh = np.array(fnc.variables['zh'])[2]
    # zh_qc = np.array(fnc.variables['zh_qc'])[2]
    # grid_zh = np.array(fnc.variables['grid_zh'])[1]
    # # zdr = np.array(fnc.variables['zdr_qc'])[2]
    # # phidp = np.array(fnc.variables['phidp_qc'])[2]
    # # cc = np.array(fnc.variables['cc_qc'])[2]
    # fnc.close()
    
    # fnc = nc.Dataset('20180716/SA/nc/Z_RADR_I_Z9010_20180716013600_O_DOR_SA_CAP.nc', 'r')
    # zh = np.array(fnc.variables['zh'])[1][:, :230]
    # fnc.close()
    
    with open('test/BJXSY.20180716.003300qc.dat', 'rb') as data_file:    
        values = np.array(struct.unpack('f'*(9*360*1000*5), data_file.read()))
    size = 9*360*1000
    i = 2
    data = values[i*size:(i+1)*size].reshape(9,360,1000)
        # values = struct.unpack('f'*(9*360*1000), data_file.read(9*360*1000))
    
    ppi(data[1], prv = 'phidp', r = np.arange(0, 75, 0.075))
    # ppi(zh_qc, prv = 'zh', r = np.arange(0, 75, 0.075))
    
    # plt.figure(figsize=(10,10))
    # plt.pcolor(grid_zh)
    # ppi(zdr, prv = 'zdr', r = np.arange(1, 1001)*0.075)
    # ppi(phidp, prv = 'phidp', r = np.arange(1, 1001)*0.075)
    # ppi(cc, prv = 'cc', r = np.arange(1, 1001)*0.075)
    
    # phidp_lp = np.load('20180716/SY/phidp_3_lp/BJXSY_20180716_000000_phidp_3_lp.npy')
    # ppi(phidp_lp, prv = 'phidp', r = np.arange(1, 1001)*0.075)
