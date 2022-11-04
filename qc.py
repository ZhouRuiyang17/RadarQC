import numpy as np
import netCDF4 as nc
import os
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from vis import *
import datetime


zh_nodata = -33
zdr_nodata = -8.125
phidp_nodata = -2
cc_nodata = -0.025

_t1 = datetime.datetime.now()

#%% non meteor
# =============================================================================
# (1) 杨文宇. 基于天气雷达的城市降雨特征及临近预报研究. 博士, 清华大学, 2017.
# 
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
'''
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
'''

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
#%% phidp QC
'''
# =============================================================================
# (1) 马建立; 陈明轩; 李思腾; 仰美霖. 线性规划在X波段双线偏振多普勒天气雷达差分传播相移质量控制中的应用. 气象学报 2019, 77 (03), 516–528.
# 先去除非气象回波，再滑动平均，最后线性规划
# =============================================================================
def smooth(phidp):
    phidp_sm = phidp.copy()
    num_ele, num_azi, num_rad = phidp_sm.shape
    
    for ie in range(num_ele):
        for ia in range(num_azi):
            y = phidp_sm[ie, ia].copy()
            # x = np.arange(len(y)).astype(np.int64)
            # x[y<-1] = np.nan
            
            window_length = 25
            y_new= savgol_filter(y[y>=0], window_length = window_length, polyorder = np.int64((window_length-1)/2))
            y[y>=0] = y_new
            
            phidp_sm[ie, ia] = y
    
    return phidp_sm
'''
# =============================================================================
# (1) Giangrande, S. E.; McGraw, R.; Lei, L. An Application of Linear Programming to Polarimetric Radar Differential Phase Processing. Journal of Atmospheric and Oceanic Technology 2013, 30 (8), 1716–1729. https://doi.org/10.1175/JTECH-D-12-00147.1.

# =============================================================================
def LP(phidp):
    from scipy import optimize
    phidp_LP = np.zeros_like(phidp)+phidp_nodata
    num_ele, num_azi, num_rad = phidp_LP.shape

    for ie in range(num_ele):
        for ia in range(num_azi):
            y = phidp[ie, ia]
            y_ori = y[y>=0]
            
            n = len(y_ori)
            # the five-point SavitzkyGolay (SG) second-order polynomial derivative filter
            m = 5 
            sg = np.array([6*(2*i-m-1)/m/(m+1)/(m-1) for i in range(1, m+1)])

            # lp
            I = np.zeros(shape=(n,n))
            for i in range(n):
                I[i,i] = 1
            A = np.vstack([np.hstack([I, -I]), np.hstack([I, I])])
            Z = np.zeros(shape=(n-m+1, n))
            M = Z.copy()
            for i in range(n-m+1):
                M[i, i:i+m] = sg
            Aaug = np.vstack([A, np.hstack([Z, M])])
            Aub = -Aaug # >= -> <=
            b = np.hstack([-y_ori, y_ori, np.zeros(n-m+1)])
            bub = -b # >= -> <=
            c = np.hstack([np.ones(n), np.zeros(n)]) # min
            res = optimize.linprog(c, Aub, bub)
            y_lp = res.x[n:]
            
            # For the five-point derivative filter, the corresponding smoothing filter is
            y_lp_sm = y_lp.copy()
            y_lp_sm[2:-2] = 0.1*y_lp[0:-4] + 0.25*y_lp[1:-3] + 0.3*y_lp[2:-2] + 0.25*y_lp[3:-1] + 0.1*y_lp[4:]
            y_lp_sm[:2], y_lp_sm[-2:]= y_lp_sm[2], y_lp_sm[-3]
            
            phidp_LP[ie, ia, np.where(y>=0)] = y_lp_sm
    
    return phidp_LP
#%% kdp calculation


#%% attenuation
# =============================================================================
# (1) 杨文宇. 基于天气雷达的城市降雨特征及临近预报研究. 博士, 清华大学, 2017.

# =============================================================================
def zh_method(zh:np.ndarray, band:str):
    if band == 's':
        zh = zh
    elif band == 'x':
        

def kdp_method(phidp_qc, band:str):
    if band == 's':
        zh = zh 
    elif band == 'x':
        

#%% main
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
        # ppi(zh[1], 'zh')
    
        mask_basic = basic_mask(zh)
        zh[mask_basic==1] = zh_nodata
        # ppi(zh[1], 'zh')
        
        mask_speck = speck(zh)
        zh[mask_speck==1] = zh_nodata
        # ppi(zh[1], 'zh')
        
        zh_method(zh, band = 's')
        # ppi(zh[1], 'zh')
    elif DP == True:
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        mask_basic = basic_mask(zh)
        zh[mask_basic==1] = zh_nodata
        zdr[mask_basic==1] = zdr_nodata
        phidp[mask_basic==1] = phidp_nodata
        cc[mask_basic==1] = cc_nodata
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        # ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        mask_nmet = nmet(zh, phidp, cc)
        zh[mask_nmet==1] = zh_nodata
        zdr[mask_nmet==1] = zdr_nodata
        phidp[mask_nmet==1] = phidp_nodata
        cc[mask_nmet==1] = cc_nodata
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        # ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        mask_speck = speck(zh)
        zh[mask_speck==1] = zh_nodata
        zdr[mask_speck==1] = zdr_nodata
        phidp[mask_speck==1] = phidp_nodata
        cc[mask_speck==1] = cc_nodata
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        phidp_qc = LP(phidp)
        ppi(phidp_qc[1], 'phidp')

    
_t2 = datetime.datetime.now()
print(_t2-_t1)