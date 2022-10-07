import numpy as np
import wradlib as wl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors
import netCDF4 as nc

def mask(data):
    data = np.array(data)
    data[data<0]=np.nan
    return data


def ppi(data, prv = None):
    
    color = np.array([[250, 250, 250], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 165, 0], 
                      [255, 0, 0], [160, 32, 140]])/255
    level = np.arange(0, 90, 10)
    cmap, norm = colors.from_levels_and_colors(level, color)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # ax, pm = wl.vis.plot_ppi(mask(data), r = np.arange(1,1001)*0.075, ax = ax, 
    #                          norm = colors.LogNorm(vmin = 0, vmax = 1), cmap = 'jet', shading = 'nearest')
    ax, pm = wl.vis.plot_ppi(mask(data), r = np.arange(1,1001)*0.075, ax = ax, cmap = cmap, norm = norm)

    cb = fig.colorbar(pm, ax=ax, label=prv)

fnc = nc.Dataset('20180716/SY/nc/BJXSY_20180716_000000.nc', 'r')
zh = np.array(fnc.variables['zh'])[2]
fnc.close()

ppi(zh)