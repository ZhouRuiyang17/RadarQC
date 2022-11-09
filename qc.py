#%% basic info
import numpy as np
import struct
from vis import *
import datetime


zh_nodata = -33
zdr_nodata = -8.125
phidp_nodata = -2
kdp_nodata = -5
cc_nodata = -0.025

_t1 = datetime.datetime.now()

#%% non-meteor mask
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
                if  area[12] > -32 and len(area[area>-32]) <= 10: # is speck? 有，但周围有的不多，则是噪点
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
    
    loc = np.where( (zh>=35) & (cc<=0.95) & (sdphidp>=10) )
    mask_nmet[loc] = 1
    loc = np.where( (zh<=35) & (cc<=0.80) & (sdphidp>=10) )
    mask_nmet[loc] = 1
    loc = np.where( (cc<=0.60) )
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
            if len(y_ori)<10:
                continue
            
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
#%% kdp reconstruction
def kdp_def(phidp_qc, reso):
    kdp_rec = np.zeros_like(phidp_qc)+kdp_nodata
    num_ele, num_azi, num_rad = kdp_rec.shape
    
    for ie in range(num_ele):
        for ia in range(num_azi):
            y = phidp_qc[ie, ia]
            loc = np.where(y>phidp_nodata)
            if len(loc[0])<10:
                continue
            y = y[loc]
            kdp = (y[1:]-y[:-1])/2/reso
            kdp_rec[ie, ia, loc[0][1:]] = kdp
            
    return kdp_rec

#%% attenuation
# =============================================================================
# (1) 杨文宇. 基于天气雷达的城市降雨特征及临近预报研究. 博士, 清华大学, 2017.

# =============================================================================
def zh_method(zh:np.ndarray, band:str):
    zh_c = zh.copy()
    
    if band == 's':
        zh_c = zh_c
    
    return zh_c    
# =============================================================================
# (1) 马玉. 基于雨滴谱和雷达观测的北京城市降雨研究.
# (1) Zhang, G. Weather Radar Polarimetry; CRC Press: Boca Raton, 2016. https://doi.org/10.1201/9781315374666.

# =============================================================================
def kdp_method(zh, zdr, phidp_qc, reso, band:str):
    zh_c = zh.copy()
    zdr_c = zdr.copy()
    
    if band == 's':
        zh_c = zh_c
        zdr_c = zdr_c
    elif band == 'x':
        # 马
        # a = 0.285
        # b = 0.026
        # Zhang
        a = 0.32
        b = 0.036
        
        num_ele, num_azi, num_rad = phidp_qc.shape
        for ie in range(num_ele):
            for ia in range(num_azi):
                y = phidp_qc[ie, ia]
                if len(y[y>0])<10:
                    continue
                phidp0 = (y[y>0])[0]    
                PIA = a*(y-phidp0)
                PIADR = b*(y-phidp0)
                for i in range(1, len(y)):
                    if PIA[i]<0:
                        PIA[i] = PIA[i-1]
                    if PIADR[i]<0:
                        PIADR[i] = PIADR[i-1]
                loc = np.where(zh_c[ie, ia]>zh_nodata)
                zh_c[ie, ia, loc] = zh_c[ie, ia, loc] + PIA[loc]
                loc = np.where(zdr_c[ie, ia]>zdr_nodata)
                zdr_c[ie, ia, loc] = zdr_c[ie, ia, loc] + PIADR[loc]
        
    elif band == 'c':
        # Zhang
        a = 0.11
        b = 0.036
        
        num_ele, num_azi, num_rad = phidp_qc.shape
        for ie in range(num_ele):
            for ia in range(num_azi):
                y = phidp_qc[ie, ia]
                if len(y[y>0])<=10:
                    continue
                phidp0 = (y[y>0])[0]    
                PIA = a*(y-phidp0)
                PIADR = b*(y-phidp0)
                for i in range(1, len(y)):
                    if PIA[i]<0:
                        PIA[i] = PIA[i-1]
                    if PIADR[i]<0:
                        PIADR[i] = PIADR[i-1]
                loc = np.where(zh_c[ie, ia]>zh_nodata)
                zh_c[ie, ia, loc] = zh_c[ie, ia, loc] + PIA[loc]
                loc = np.where(zdr_c[ie, ia]>zdr_nodata)
                zdr_c[ie, ia, loc] = zdr_c[ie, ia, loc] + PIADR[loc]
    
        
    
        
    return zh_c, zdr_c
#%% qc
def qc(fpath, new_fpath, band, reso, dp):
  
    if dp:
        with open(fpath, 'rb') as data_file:    
            values = np.array(struct.unpack('f'*(9*360*1000*5), data_file.read()))
        size = 9*360*1000
        zh = values[:size].reshape(9,360,1000)
        zdr = values[size:2*size].reshape(9,360,1000)
        phidp = values[2*size:3*size].reshape(9,360,1000)
        # kdp = values[3*size:4*size].reshape(9,360,1000)
        cc = values[4*size:5*size].reshape(9,360,1000)

    else:
        with open(fpath, 'rb') as data_file:    
            values = np.array(struct.unpack('f'*(9*360*1000), data_file.read()))
        size = 9*360*1000
        zh = values[:size].reshape(9,360,1000)


    if dp == False:
# =============================================================================
#         qc
# =============================================================================
        # ppi(zh[1], 'zh')
    
        mask_basic = basic_mask(zh)
        zh[mask_basic==1] = zh_nodata
        # ppi(zh[1], 'zh')
        
        mask_speck = speck(zh)
        zh[mask_speck==1] = zh_nodata
        # ppi(zh[1], 'zh')
        
        zh = zh_method(zh, band = 's')
        # ppi(zh[1], 'zh')
# =============================================================================
#         save
# =============================================================================
        with open(new_fpath, 'wb') as new_file:
            new_file.write(struct.pack('f'*len(zh.flatten()), *zh.flatten()))
           
    elif dp == True:
# =============================================================================
#         qc
# =============================================================================
        # ppi(zh[1], 'zh')
        # ppi(zdr[1], 'zdr')
        # ppi(phidp[1], 'phidp')
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
        # ppi(phidp[1], 'phidp')
        # ppi(cc[1], 'cc')
        
        phidp = LP(phidp)
        # ppi(phidp_qc[1], 'phidp')
        
        kdp = kdp_def(phidp, reso)
        # ppi(kdp_rec[1],'kdp')
        
        zh, zdr = kdp_method(zh, zdr, phidp, reso, band)
        # ppi(zh_c[1], 'zh')
        # ppi(zdr_c[1], 'zdr')
# =============================================================================
#         save
# =============================================================================
        with open(new_fpath, 'wb') as new_file:
            new_file.write(struct.pack('f'*len(zh.flatten()), *zh.flatten()))
            new_file.write(struct.pack('f'*len(zdr.flatten()), *zdr.flatten()))
            new_file.write(struct.pack('f'*len(phidp.flatten()), *phidp.flatten()))
            new_file.write(struct.pack('f'*len(kdp.flatten()), *kdp.flatten()))
            new_file.write(struct.pack('f'*len(cc.flatten()), *cc.flatten()))
    
