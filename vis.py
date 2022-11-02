import numpy as np
import wradlib as wl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import netCDF4 as nc


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
        level = np.arange(-7, 9, 1)
        cb_label = '$Z_{DR}$'
    elif prv == 'phidp':
        level = np.arange(0, 160, 10)
        cb_label = '$\phi_{DP}$ ($\degree$)'
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




if __name__ == "__main__":
    # fnc = nc.Dataset('20180716/SY/nc/BJXSY_20180716_000000.nc', 'r')
    # zh = np.array(fnc.variables['zh'])[2]
    # zdr = np.array(fnc.variables['zdr'])[2]
    # phidp = np.array(fnc.variables['phidp'])[2]
    # cc = np.array(fnc.variables['cc'])[2]
    # fnc.close()
    
    fnc = nc.Dataset('20180716/SA/nc/Z_RADR_I_Z9010_20180716013600_O_DOR_SA_CAP.nc', 'r')
    zh = np.array(fnc.variables['zh'])[1][:, :230]
    fnc.close()
    
    ppi(zh, prv = 'zh', r = np.arange(1, 231)*1)
    # ppi(zdr, prv = 'zdr', r = np.arange(1, 1001)*0.075)
    # ppi(phidp, prv = 'phidp', r = np.arange(1, 1001)*0.075)
    # ppi(cc, prv = 'cc', r = np.arange(1, 1001)*0.075)
    
    # phidp_lp = np.load('20180716/SY/phidp_3_lp/BJXSY_20180716_000000_phidp_3_lp.npy')
    # ppi(phidp_lp, prv = 'phidp', r = np.arange(1, 1001)*0.075)
