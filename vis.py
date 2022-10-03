import numpy as np
import wradlib as wl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs



def mask(data):
    data = np.array(data)
    data[data<0]=np.nan
    return data


def ppi(data, prv: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax, pm = wl.vis.plot_ppi(mask(data), r=np.arange(1,1001)*0.075, ax=ax, cmap='jet')
    cb = fig.colorbar(pm, ax=ax, label=prv)
    
phidp_3_lp = np.load('20180716/SY/phidp_3_lp/BJXSY_20180716_000000_phidp_3_lp.npy')
phidp_3 = np.load('20180716/SY/npy/BJXSY_20180716_000000_phidp.npy')[2]
ppi(phidp_3, prv='$\phi_{dp}$')
ppi(phidp_3_lp, prv='$\phi_{dp}$')