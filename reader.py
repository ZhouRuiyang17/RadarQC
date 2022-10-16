import os
import numpy as np
import netCDF4 as nc
import warnings
from typing import Tuple
from collections import OrderedDict
warnings.filterwarnings("ignore")

from vis import *
#%%

def metstar_readar(dta_path):
    '''
    reader for MetSTAR (敏视达)
        finished by Xuejian and Ruidong 
    '''

    
    
    variable_encode = {1: "dbt", 2: "dbz", 7: "zdr", 9: "cc", 10: "phidp", 11: "kdp"}

    # [1] Read raw data produced by Beijing METSTAR radar Co.,LTD.
    fid = open(dta_path, "rb")
    # Shift value for file pointers of different block
    # ---generic header block pointer (len=32)
    pt_common_block = 0
    # ---site configuration information block pointer (len=128)
    pt_site_block = 0 + 32
    # ---task configuration information block pointer (len=256)
    pt_task_block = 0 + 32 + 128
    # ---1st angle of elevation block pointer
    # ------The shift value of N-th angle of elevation is: 416 + (N-1) * 256
    pt_1st_ele = 0 + 32 + 128 + 256

    siteinfo = {}
    taskinfo = {}
    eleinfo = {}
    radinfo = {}
    f = {'radinfo':{},'dbt':{},'dbz':{},'zdr':{},'cc':{},'phidp':{},'kdp':{}}

    # ---read site configuration information
    fid.seek(pt_site_block, 0)
    # ------site code
    siteinfo['code'] = ''.join([chr(item) for item in np.fromfile(file=fid, dtype=np.int8, count=8)])
    # ------site name
    siteinfo['name'] = ''.join([chr(item) for item in np.fromfile(fid, np.int8, 32)])
    # ------latitude
    siteinfo['lat'] = np.fromfile(fid, np.float32, 1)[0]
    # ------longitude
    siteinfo['lon'] = np.fromfile(fid, np.float32, 1)[0]
    # ------antenna height
    siteinfo['atennaasl'] = np.fromfile(fid, np.int32, 1)[0]
    # ------ground height
    siteinfo['baseasl'] = np.fromfile(fid, np.int32, 1)[0]
    # ------frequency
    siteinfo['freq'] = np.fromfile(fid, np.float32, 1)[0]
    # ------beam width (Horizontal)
    siteinfo['beamhwidth'] = np.fromfile(fid, np.float32, 1)[0]
    # ------beam width (Vertical)
    siteinfo['beamvwidth'] = np.fromfile(fid, np.float32, 1)[0]

    # ---read task configuration information
    fid.seek(pt_task_block, 0)
    # ------task name
    taskinfo['name'] = ''.join([chr(item) for item in np.fromfile(fid, np.int8, 32)])
    # ------task description
    taskinfo['description'] = ''.join([chr(item) for item in np.fromfile(fid, np.int8, 128)])
    # ------polarization type
    taskinfo['polmode'] = np.fromfile(fid, np.int32, 1)[0]
    # ------scan type: 1 for Plan Position Indicator (PPI), 2 for Range Height Indicator (RHI)
    taskinfo['scantype'] = np.fromfile(fid, np.int32, 1)[0]
    # ------pulse width
    taskinfo['pulsewidth'] = np.fromfile(fid, np.int32, 1)[0]
    # ------!!! scan start time (UTC, start from 1970/01/01 00:00)
    taskinfo['startime'] = np.fromfile(fid, np.int32, 1)[0]
    # ------!!! cut number
    taskinfo['cutnum'] = np.fromfile(fid, np.int32, 1)[0]
    # ------horizontal noise
    taskinfo['hnoise'] = np.fromfile(fid, np.float32, 1)[0]
    # ------vertical noise
    taskinfo['vnoise'] = np.fromfile(fid, np.float32, 1)[0]
    # ------horizontal calibration
    taskinfo['hsyscal'] = np.fromfile(fid, np.float32, 1)[0]
    # ------vertical calibration
    taskinfo['vsyscal'] = np.fromfile(fid, np.float32, 1)[0]
    # ------horizontal noise temperature
    taskinfo['hte'] = np.fromfile(fid, np.float32, 1)[0]
    # ------vertical noise temperature
    taskinfo['vte'] = np.fromfile(fid, np.float32, 1)[0]
    # ------ZDR calibration
    taskinfo['zdrbias'] = np.fromfile(fid, np.float32, 1)[0]
    # ------PhiDP calibration
    taskinfo['phasebias'] = np.fromfile(fid, np.float32, 1)[0]
    # ------LDR calibration
    taskinfo['ldrbias'] = np.fromfile(fid, np.float32, 1)[0]

    # ---read angle of elevation block
    for ct in range(1, int(taskinfo['cutnum'])+1):
        fid.seek(pt_1st_ele + (ct-1) * 256, 0)
        info_dict = {}
        # ------process mode
        info_dict['mode'] = np.fromfile(fid, np.int32, 1)[0]
        # ------wave form
        info_dict['waveform'] = np.fromfile(fid, np.int32, 1)[0]
        # ------pulse repetition frequency 1 (PRF1)
        info_dict['prf1'] = np.fromfile(fid, np.float32, 1)[0]
        # ------pulse repetition frequency 2 (PRF2)
        info_dict['prf2'] = np.fromfile(fid, np.float32, 1)[0]
        # ------de-aliasing mode
        info_dict['unfoldmode'] = np.fromfile(fid, np.int32, 1)[0]
        # ------azimuth for RHI mode
        info_dict['azi'] = np.fromfile(fid, np.float32, 1)[0]
        # ------elevation for PPI mode
        info_dict['ele'] = np.fromfile(fid, np.float32, 1)[0]
        # ------start angle, i.e., start azimuth for PPI mode or highest elevation for RHI mode
        info_dict['startangle'] = np.fromfile(fid, np.float32, 1)[0]
        # ------end angle, i.e., end azimuth for PPI mode or lowest elevation for RHI mode
        info_dict['endangle'] = np.fromfile(fid, np.float32, 1)[0]
        # ------angular resolution (only for PPI mode)
        info_dict['angleres'] = np.fromfile(fid, np.float32, 1)[0]
        # ------scan speed
        info_dict['scanspeed'] = np.fromfile(fid, np.float32, 1)[0]
        # ------log resolution
        info_dict['logres'] = np.fromfile(fid, np.int32, 1)[0]
        # ------Doppler Resolution
        info_dict['dopres'] = np.fromfile(fid, np.int32, 1)[0]
        # ------Maximum Range corresponding to PRF1
        info_dict['maxrange1'] = np.fromfile(fid, np.int32, 1)[0]
        # ------Maximum Range corresponding to PRF2
        info_dict['maxrange2'] = np.fromfile(fid, np.int32, 1)[0]
        # ------start range
        info_dict['startrange'] = np.fromfile(fid, np.int32, 1)[0]
        # ------number of samples corresponding to PRF1
        info_dict['samplenum1'] = np.fromfile(fid, np.int32, 1)[0]
        # ------number of samples corresponding to PRF2
        info_dict['samplenum2'] = np.fromfile(fid, np.int32, 1)[0]
        # ------phase mode
        info_dict['phasemode'] = np.fromfile(fid, np.int32, 1)[0]
        # ------atmosphere loss
        info_dict['atmosloss'] = np.fromfile(fid, np.float32, 1)[0]
        # ------Nyquist speed
        info_dict['vmax'] = np.fromfile(fid, np.float32, 1)[0]
        # ------moments mask
        info_dict['mask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------moments size mask
        info_dict['masksize'] = np.fromfile(fid, np.float32, 1)[0]
        info_dict['datasizemask'] = [mas for mas in np.fromfile(fid, np.float64, 64)]
        # ------misc filter mask
        info_dict['filtermask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------SQI threshold
        info_dict['sqi'] = np.fromfile(fid, np.float32, 1)[0]
        # ------SIG threshold
        info_dict['sig'] = np.fromfile(fid, np.float32, 1)[0]
        # ------CSR threshold
        info_dict['csr'] = np.fromfile(fid, np.float32, 1)[0]
        # ------LOG threshold
        info_dict['log'] = np.fromfile(fid, np.float32, 1)[0]
        # ------CPA threshold
        info_dict['cpa'] = np.fromfile(fid, np.float32, 1)[0]
        # ------PMI threshold
        info_dict['pmi'] = np.fromfile(fid, np.float32, 1)[0]
        # ------reserved threshold
        info_dict['threshold'] = np.fromfile(fid, np.int8, 8)[0]
        # ------dBT threshold
        info_dict['dbtmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------dBZ mask
        info_dict['dbzmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------velocity mask
        info_dict['vmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------spectrum width mask
        info_dict['wmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------DP mask
        info_dict['zdrmask'] = np.fromfile(fid, np.int32, 1)[0]
        # ------mask reserved
        info_dict['maskreserved'] = np.fromfile(fid, np.int8, 12)
        # ------flag for scan synchronization
        info_dict['scansync'] = np.fromfile(fid, np.int32, 1)[0]
        # ------scan direction
        info_dict['scandirection'] = np.fromfile(fid, np.int32, 1)[0]
        # ------ground clutter classifier type
        info_dict['cmap'] = np.fromfile(fid, np.int16, 1)[0]
        # ------ground clutter filter type
        info_dict['cfiltertype'] = np.fromfile(fid, np.int16, 1)[0]
        # ------ground clutter filter notch width
        info_dict['cnotchwidth'] = np.fromfile(fid, np.int16, 1)[0]
        # ------ground clutter filter window
        info_dict['cfilterwin'] = np.fromfile(fid, np.int16, 1)[0]
        # ------reserved
        info_dict['twin'] = np.fromfile(fid, np.int8, 1)[0]
        eleinfo[str(ct)] = info_dict

    # ---read radial data block
    pt_radial_block = pt_1st_ele + int(taskinfo['cutnum']) * 256
    fid.seek(pt_radial_block, 0)
    a = 0
    # ------we have 12 angle of elevation
    for i in range(1, 12):
        radinfo[str(i)] = {}
        # ------radial state
        radinfo[str(i)]['state'] = []
        # ------spot blank
        radinfo[str(i)]['spotblank'] = []
        # ------sequence number
        radinfo[str(i)]['seqnum'] = []
        # ------radial number
        radinfo[str(i)]['curnum'] = []
        # ------elevation number
        radinfo[str(i)]['elenum'] = []
        # ------azimuth
        radinfo[str(i)]['azi'] = []
        # ------elevation
        radinfo[str(i)]['ele'] = []
        # ------seconds
        radinfo[str(i)]['sec'] = []
        # ------microseconds
        radinfo[str(i)]['micro'] = []
        # ------length of data
        radinfo[str(i)]['datalen'] = []
        # ------moment number
        radinfo[str(i)]['momnum'] = []
        radinfo[str(i)]['reserved'] = []

    for i in range(1, 12):
        f['dbt'][i] = {}
        f['dbz'][i] = {}
        f['zdr'][i] = {}
        f['cc'][i] = {}
        f['phidp'][i] = {}
        f['kdp'][i] = {}

    while True:
        # ------radial state
        state = np.fromfile(fid, np.int32, 1)[0]
        # ------spot blank
        spotblank = np.fromfile(fid, np.int32, 1)[0]
        # ------sequence number
        seqnum = np.fromfile(fid, np.int32, 1)[0]
        # ------radial number
        curnum = np.fromfile(fid, np.int32, 1)[0]
        # ------elevation number
        elenum = np.fromfile(fid, np.int32, 1)[0]

        radinfo[str(int(elenum))]['state'].append(state)
        radinfo[str(int(elenum))]['spotblank'].append(spotblank)
        radinfo[str(int(elenum))]['seqnum'].append(seqnum)
        radinfo[str(int(elenum))]['curnum'].append(curnum)
        radinfo[str(int(elenum))]['elenum'].append(elenum)

        # ------azimuth
        radinfo[str(int(elenum))]['azi'].append(np.fromfile(fid, np.float32, 1)[0])
        # ------elevation
        radinfo[str(int(elenum))]['ele'].append(np.fromfile(fid, np.float32, 1)[0])
        # ------seconds
        radinfo[str(int(elenum))]['sec'].append(np.fromfile(fid, np.int32, 1)[0])
        # ------microseconds
        radinfo[str(int(elenum))]['micro'].append(np.fromfile(fid, np.int32, 1)[0])
        # ------length of data
        datalen = np.fromfile(fid, np.int32, 1)[0]
        radinfo[str(int(elenum))]['datalen'].append(datalen)
        # ------moment number
        num_moment = np.fromfile(fid, np.int32, 1)[0]
        radinfo[str(int(elenum))]['momnum'].append(num_moment)
        radinfo[str(int(elenum))]['reserved'].append(np.fromfile(fid, np.int8, 20))

        radcnt = curnum
        b = 0
        for n in range(0, num_moment):
            # ------data type
            var_type = np.fromfile(fid, np.int32, 1)[0]
            # ------scale
            scale = np.fromfile(fid, np.int32, 1)[0]
            # ------offset
            offset = np.fromfile(fid, np.int32, 1)[0]
            # ------!!! bin length: bytes for one bin storage
            binbytenum = np.fromfile(fid, np.int16, 1)[0]
            # ------flags
            flag = np.fromfile(fid, np.int16, 1)[0]
            # ------length
            bin_length = np.fromfile(fid, np.int32, 1)[0]
            reserved = np.fromfile(fid, np.int8, 12)

            bin_num = bin_length / binbytenum

            if binbytenum == 1:
                data_raw = np.fromfile(fid, np.uint8, int(bin_num))
            else:
                data_raw = np.fromfile(fid, np.uint16, int(bin_num))

            b += 1

            if elenum != 2 and elenum != 4:
                if var_type in variable_encode.keys():
                    # Note: When operated between unsigned int8 and int32, numpy might give some mis-conversion
                    f[variable_encode[var_type]][int(elenum)][str(int(radcnt))] = (1.0 * data_raw - offset)/scale
                    # f[str][int][str]
        a += 1
        if state == 6 or state == 4:
            break

    return siteinfo, taskinfo, eleinfo, radinfo, f

def metstar_saver(save_path, save_name, site, task, ele, rad, data, num_ele = 11, num_rad = 360, num_gate = 1000):
    '''
    saver for MetSTAR (敏视达) in '.nc'
        finished by Ruiyang
    '''
    
    zh = np.zeros(shape=(num_ele,num_rad,num_gate))
    zdr = np.zeros(shape=(num_ele,num_rad,num_gate))
    phidp = np.zeros(shape=(num_ele,num_rad,num_gate))
    cc = np.zeros(shape=(num_ele,num_rad,num_gate))
    
    # 20210527, learn from Zhang Zhe
    ele = 0
    for ele_id in range(1,12):
        flg1 = data['dbz'][ele_id]
        flg2 = data['phidp'][ele_id]
        flg3 = data['cc'][ele_id]
        # 如果没有数据，则跳过此仰角
        if flg1 == {} or flg2 == {} or flg3 == {} :
            ele += 1
            continue
        
        
        # 获取方位角
        azi = np.array(rad[str(ele_id)]['azi'])
        # 对方位角排序
        rank = np.argsort(azi)
        # 四舍五入取整方位角
        azi = np.round(azi[rank]).astype(np.int)
        # 计数器：应该有360个方位角
        c = np.zeros(num_rad)
        # 先获取数据
        zh_id = np.array([data['dbz'][ele_id][str(i)] for i in range(1, len(azi)+1)])[:,:num_gate]
        zdr_id = np.array([data['zdr'][ele_id][str(i)] for i in range(1, len(azi)+1)])[:,:num_gate]
        phidp_id = np.array([data['phidp'][ele_id][str(i)] for i in range(1, len(azi)+1)])[:,:num_gate]
        cc_id = np.array([data['cc'][ele_id][str(i)] for i in range(1, len(azi)+1)])[:,:num_gate]
        # 按照方位角进行排序
        zh_id = zh_id[rank]
        zdr_id = zdr_id[rank]
        phidp_id = phidp_id[rank]
        cc_id = cc_id[rank]
        # 安排数据
        for i in range(0, len(azi)):
            radial = azi[i]
            # 对于第i个方位角，如果i-1是空的，则不仅填自己i，也填到i-1
            if c[radial-1] == 0:
                zh[ele,radial-1,:] = zh_id[i]
                zdr[ele,radial-1,:] = zdr_id[i]
                phidp[ele,radial-1,:] = phidp_id[i]
                cc[ele,radial-1,:] = cc_id[i]
                # 因为i-1被填上，所以就计数+1
                c[radial-1] += 1
            # 第360个方位角就是第0个方位角
            if radial == 360:
                radial = 0
            # 填到第i个仰角
            zh[ele,radial,:] = zh_id[i]
            zdr[ele,radial,:] = zdr_id[i]
            phidp[ele,radial,:] = phidp_id[i]
            cc[ele,radial,:] = cc_id[i]
            # 因为i被填上，所以就计数+1
            c[radial] += 1
        ele += 1
   
    fnc = nc.Dataset(save_path+save_name+'.nc', 'a')
    fnc.site_name = site['name']
    fnc.site_code = site['code']
    fnc.lon = site['lon']
    fnc.lat = site['lat']
    fnc.h_base = site['baseasl']
    fnc.h_atenna = site['atennaasl']
    fnc.beam_width_h = site['beamhwidth']
    fnc.beam_width_v = site['beamvwidth']
    fnc.freq = site['freq']
    fnc.elevation = np.array([0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 6.00, 9.90, 14.60, 19.50])
    fnc.spatial_reso = 0.075
    
    fnc.createDimension('ele', num_ele)
    fnc.createDimension('azi', num_rad)
    fnc.createDimension('gate', num_gate)
    
    fnc.createVariable('zh', np.float64, ('ele', 'azi', 'gate'))
    fnc.variables['zh'][:] = zh
    fnc.createVariable('zdr', np.float64, ('ele', 'azi', 'gate'))
    fnc.variables['zdr'][:] = zdr
    fnc.createVariable('phidp', np.float64, ('ele', 'azi', 'gate'))
    fnc.variables['phidp'][:] = phidp
    fnc.createVariable('cc', np.float64, ('ele', 'azi', 'gate'))
    fnc.variables['cc'][:] = cc

    fnc.close()
    
def metstar(path, save_path, f):
    site, task, ele, rad, data=metstar_readar(path+f)
    save_name = "_".join(f.split(".")[:-1])
    metstar_saver(save_path, save_name, site, task, ele, rad, data)


#%%
AZIMUTH_RANGE = 360
MAX_NUM_REF_RANGE_BIN = 460
MAX_NUM_DOP_RANGE_BIN = 920
DEFAULT_REF = -33.0
DEFAULT_DOP = -64.5

def elevation_mapping(elevation: float) -> float:
    if elevation < 1.0:
        elevation = 0.5
    elif elevation >= 1.0 and elevation < 2.0:
        elevation = 1.5
    elif elevation >= 2.0 and elevation < 3.0:
        elevation = 2.4
    elif elevation >= 3.0 and elevation < 4.0:
        elevation = 3.4
    elif elevation >= 4.0 and elevation < 5.0:
        elevation = 4.3
    elif elevation >= 5.0 and elevation < 8.0:
        elevation = 6.0
    elif elevation > 8.0 and elevation < 12.0:
        elevation = 9.9
    elif elevation >= 12 and elevation < 18.0:
        elevation = 14.6
    elif elevation >= 18.0:
        elevation = 19.5
    return elevation


def azimuth_mapping(azimuth: float) -> float:
    azimuth = float(round(azimuth))
    if azimuth >= AZIMUTH_RANGE:
        azimuth = azimuth - AZIMUTH_RANGE
    return azimuth


def read_radar_bin(path: str) -> Tuple[np.ndarray, np.ndarray]:
    size = os.path.getsize(path)
    num_scans = size // 2432
    
    with open(path, 'rb') as f:
        total_data = {}
        
        for i in range(num_scans):
            # Header
            f.seek(28, 1)

            # Basic information
            milliseconds = int.from_bytes(f.read(4), 'little')
            days = int.from_bytes(f.read(2), 'little')
            unambiguous_distance = int.from_bytes(f.read(2), 'little') / 10.0

            # Crucial information
            azimuth = int.from_bytes(f.read(2), 'little') / 8.0 * 180.0 / 4096.0
            azimuth = azimuth_mapping(azimuth)
            radial_order = int.from_bytes(f.read(2), 'little')
            radial_status = int.from_bytes(f.read(2), 'little')
            elevation = int.from_bytes(f.read(2), 'little') / 8.0 * 180.0 / 4096.0
            elevation = elevation_mapping(elevation)
            num_elevations = int.from_bytes(f.read(2), 'little')
            f.seek(8, 1)
            # ref_1st_range_bin = int.from_bytes(f.read(2), 'little')
            # dop_1st_range_bin = int.from_bytes(f.read(2), 'little')
            # ref_range_bin_distance = int.from_bytes(f.read(2), 'little')
            # dop_range_bin_distance = int.from_bytes(f.read(2), 'little')
            num_ref_range_bin = int.from_bytes(f.read(2), 'little')
            num_dop_range_bin = int.from_bytes(f.read(2), 'little')
            f.seek(4, 1)
            # num_sector = int.from_bytes(f.read(2), 'little')
            # correction_coefficient = int.from_bytes(f.read(4), 'little')
            f.seek(6, 1)
            # ref_pointer = int.from_bytes(f.read(2), 'little')
            # dop_pointer = int.from_bytes(f.read(2), 'little')
            # width_pointer = int.from_bytes(f.read(2), 'little')
            f.seek(4, 1)
            # dop_speed_res = int.from_bytes(f.read(2), 'little') / 4.0
            # vcp_mode = int.from_bytes(f.read(2), 'little')
            f.seek(8, 1)
            f.seek(6, 1)
            # ref_rev_pointer = int.from_bytes(f.read(2), 'little')
            # dop_rev_pointer = int.from_bytes(f.read(2), 'little')
            # width_rev_pointer = int.from_bytes(f.read(2), 'little')
            nyquist_speed = int.from_bytes(f.read(2), 'little') / 100.0
            f.seek(38, 1)

            if radial_order == 1:
                if not elevation in total_data.keys():
                    total_data[elevation] = {}
                    for a in range(AZIMUTH_RANGE):
                        total_data[elevation][a] = np.ones(MAX_NUM_REF_RANGE_BIN) * DEFAULT_REF

            # Reflectivity
            refs = np.ones(MAX_NUM_REF_RANGE_BIN) * DEFAULT_REF
            if num_ref_range_bin > 0:
                for n in range(num_ref_range_bin):
                    ref = (int.from_bytes(f.read(1), 'little') - 2) / 2 - 32.0
                    refs[n] = ref
                total_data[elevation][azimuth] = refs
            
            # Doppler speed
            f.seek(num_dop_range_bin, 1)
            
            # Spectrual width
            f.seek(num_dop_range_bin, 1)
            
            # Tail
            pointer = f.tell()
            f.seek(2432 * (i + 1) - pointer, 1)

        ordered_total_data = {}
        for e in total_data.keys():
            ordered_total_data[e] = np.array(list(OrderedDict(sorted(total_data[e].items(), key=lambda x: x[0])).values()))

        elevations = np.array(list(ordered_total_data.keys()))
        reflectivities = np.stack(list(ordered_total_data.values()))

    return elevations, reflectivities    

def sa_saver(save_path, save_name, zh, num_ele = 9, num_rad = 360, num_gate = 460):
    fnc = nc.Dataset(save_path+save_name+'.nc', 'a')
    fnc.elevation = np.array([0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 6.00, 9.90, 14.60, 19.50])
    fnc.spatial_reso = 1
    fnc.site_name = 'Beijing'
    fnc.site_code = save_name[9:14]
    
    fnc.createDimension('ele', num_ele)
    fnc.createDimension('azi', num_rad)
    fnc.createDimension('gate', num_gate)
    
    fnc.createVariable('zh', np.float64, ('ele', 'azi', 'gate'))
    fnc.variables['zh'][:] = zh
    
    fnc.close()

    
def sa(path, save_path, f):
    _, zh_9 = read_radar_bin(path+f)
    save_name = f[:-4]
    sa_saver(save_path, save_name, zh_9)

#%%
if __name__ == "__main__":
    path = "20180716/SA/"
    files = os.listdir(path)
    save_path = 'nc'
    if save_path not in files:
        save_path = os.mkdir(path+'nc')
        print('made new save_path')
    else:
        save_path = path+'nc/'
        print('save_path is made')
    
    for f in files[2:]:
        # if f.endswith(".AR2"):
            # metstar(path, save_path, f)
        if f.endswith('.bin'):
            sa(path, save_path, f)