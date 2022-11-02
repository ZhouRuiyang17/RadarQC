import numpy as np
import netCDF4 as nc
import os
from scipy.optimize import curve_fit

from vis import *
import datetime
_t1 = datetime.datetime.now()

#%%
# =============================================================================
# (1) 杨文宇. 基于天气雷达的城市降雨特征及临近预报研究. 博士, 清华大学, 2017.

# =============================================================================
def basic_mask(zh):
    mask_basic = np.zeros_like(zh)
    loc = np.where(zh<=5)
    mask_basic[loc] = 1
    return mask_basic

# =============================================================================
# (1) Ośródka, K.; Szturc, J.; Jurczyk, A. Chain of Data Quality Algorithms for 3-D Single-Polarization Radar Reflectivity (RADVOL-QC System): Data Quality Algorithms for 3-D Single-Polarization Radar Reflectivity. Met. Apps 2014, 21 (2), 256–270. https://doi.org/10.1002/met.1323.

# =============================================================================
def speck(zh:np.ndarray):
    mask_speck = np.zeros_like(zh)
    num_ele, num_azi, num_rad = zh.shape
    
    for ie in range(num_ele):
        zhi = np.vstack([zh[ie, -2:], zh[ie]])
        zhi = np.vstack([zhi, zh[ie, :2]])
        
        for j in range(2, num_azi+2):
            for k in range(2, num_rad-2):
                ia = j-2
                ir = k
                
                area = zhi[j-2:j+3, k-2:k+3].reshape(-1)
                if  area[12] > -32 and len(area[area>-32]) <= 5: # is speck? 有，但周围有的不多，则是噪点
                    mask_speck[ie, ia, ir] = 1

        
    return mask_speck
        
# =============================================================================
# (1) 马建立; 陈明轩; 李思腾; 仰美霖. 线性规划在X波段双线偏振多普勒天气雷达差分传播相移质量控制中的应用. 气象学报 2019, 77 (03), 516–528.

# =============================================================================
# def nmet(zdr, phidp, cc):
#     mask_nmet, sdzdr, sdphidp = np.zeros_like(cc), np.zeros_like(cc), np.zeros_like(cc)
#     num_ele, num_azi, num_rad = mask_nmet.shape
    
#     for ie in range(num_ele):
#         for ir in range(3, num_rad-3):
#             sdzdr[ie, :, ir] = np.std(zdr[ie, :, ir-3:ir+4], axis = 1)
#             sdphidp[ie, :, ir] = np.std(phidp[ie, :, ir-3:ir+4], axis = 1)
#     loc = np.where( (sdzdr>1) & (sdphidp>5) & (cc<0.9) )
#     mask_nmet[loc] = 1
#     plt.pcolor(sdzdr[1])

#     return mask_nmet


# =============================================================================
# (1) Ośródka, K.; Szturc, J. Improvement in Algorithms for Quality Control of Weather Radar Data (RADVOL-QC System). Atmos. Meas. Tech. 2022, 15 (2), 261–277. https://doi.org/10.5194/amt-15-261-2022.

# =============================================================================
def nmet(zh, phidp, cc):
    mask_nmet, sdphidp = np.zeros_like(cc), np.zeros_like(cc)
    num_ele, num_azi, num_rad = mask_nmet.shape
    
    for ie in range(num_ele):
        for ir in range(3, num_rad-3):
            sdphidp[ie, :, ir] = np.std(phidp[ie, :, ir-3:ir+4], axis = 1)
    loc = np.where( (zh>=35) & (sdphidp>=10) & (cc<=0.95) )
    mask_nmet[loc] = 1
    loc = np.where( (zh<=35) & (sdphidp>=10) & (cc<=0.80) )
    mask_nmet[loc] = 1

    return mask_nmet
#%%



#%%
# def fun(x, k, b):
#     return 2*k*x+b
# def calculate_kdp(phidp:np.ndarray):
#     num_ele, num_azi, num_rad = phidp.shape
    
#     kdp = np.zeros_like(phidp)-5
#     for ie in range(num_ele):
#         for ia in range(num_azi):
#             loc = np.where(phidp[ie, ia]>-5)[0]
#             if len(loc) >= 10:
                
#                 i = 5
#                 p = loc[i]
#                 while p<num_rad-5:
#                     y = phidp[ie, ia, p-5:p+5]
#                     x = np.arange(len(y))*0.075
#                     if np.min(y) > -5: # 连续
#                         para, _ = curve_fit(fun, x, y)
#                         kdp[ie, ia, p] = para[0]
#                     i = i+1
#                     if i>=len(loc):
#                         break
#                     else:
#                         p = loc[i]
#     return kdp

#%%
def zh_method(zh:np.ndarray, band:str):
    if band == 'S':
        zh = zh




#%%
if __name__ == '__main__':
    # fname = '20180716/SA/nc/Z_RADR_I_Z9010_20180716000000_O_DOR_SA_CAP.nc'
    # fname = '20180716/SA/nc/Z_RADR_I_Z9010_20180716004201_O_DOR_SA_CAP.nc'
    fname = '20180716/SY/nc/BJXSY_20180716_000000.nc'

    if 'SA' in fname:
        band = 's'
        if 'SAD' in fname:
            DP = True
        else:
            DP = False
    elif 'BJX' in fname:
        band = 'x'
        DP = True
    
    f = nc.Dataset(fname, 'r')
    if DP:
        zh = np.array(f.variables['zh'])[[0,2,4,5,6,7,8,9,10]]
        zdr = np.array(f.variables['zdr'])[[0,2,4,5,6,7,8,9,10]]
        phidp = np.array(f.variables['phidp'])[[0,2,4,5,6,7,8,9,10]]
        cc = np.array(f.variables['cc'])[[0,2,4,5,6,7,8,9,10]]
    else:
        zh = np.array(f.variables['zh'])
    f.close()


    if DP == False:
        ppi(zh[1], 'zh')
    
        mask_basic = basic_mask(zh)
        zh[mask_basic==1] = -33
        ppi(zh[1], 'zh')
        
        mask_speck = speck(zh)
        zh[mask_speck==1] = -33
        ppi(zh[1], 'zh')
        
        zh_method(zh, band = 's')
        ppi(zh[1], 'zh')
    elif DP == True:
        ppi(zh[1], 'zh')
        ppi(zdr[1], 'zdr')
        ppi(phidp[1], 'phidp')
        ppi(cc[1], 'cc')
        
        mask_basic = basic_mask(zh)
        zh[mask_basic==1] = -33
        zdr[mask_basic==1] = -8.125
        phidp[mask_basic==1] = -2
        cc[mask_basic==1] = -0.025
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        # ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        mask_nmet = nmet(zh, phidp, cc)
        zh[mask_nmet==1] = -33
        zdr[mask_nmet==1] = -8.125
        phidp[mask_nmet==1] = -2
        cc[mask_nmet==1] = -0.025
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        # ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        mask_speck = speck(zh)
        zh[mask_speck==1] = -33
        zdr[mask_speck==1] = -8.125
        phidp[mask_speck==1] = -2
        cc[mask_speck==1] = -0.025
        ppi(zh[1], 'zh')
        ppi(zdr[1], 'zdr')
        ppi(phidp[1], 'phidp')
        ppi(cc[1], 'cc')
    
_t2 = datetime.datetime.now()
print(_t2-_t1)