# import vis
import unzip
import read
import qc
# import grid

import os
import datetime

if __name__ == '__main__':
    path = '20180716/FS'
    save_path = 'out2'
    ls = os.listdir(path)
    for fname in ls[:]:
        if fname.endswith('bz2'):
            print(fname)
            time1 = datetime.datetime.now()
            
            fpath = os.path.join(path, fname)
            new_fpath = os.path.join(save_path, fname[:-4])
            
            unzip.unzip_bz2(fpath, new_fpath)
    
            if 'BJX' in fname:
                read.metstar(new_fpath, new_fpath[:-4]+'.dat')
                band = 'x'
                reso = 0.075
                dp = True
            
            qc.qc(new_fpath[:-4]+'.dat', new_fpath[:-4]+'qc.dat', band = band, reso = reso, dp = dp)
            
            time2 = datetime.datetime.now()
            print(time2-time1)