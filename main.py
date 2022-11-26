# import vis
import unzip
import read
import qc
# import grid

import os
import datetime
import struct

if __name__ == '__main__':
    path = 'test'
    save_path = 'test'
    ls = os.listdir(path)
    for fname in ls[:]:
        if fname.endswith('bz2'):
            print(fname)
            time1 = datetime.datetime.now()
            
            fpath = os.path.join(path, fname)
            fname = '.'.join(fname.split('.')[:-1])
            new_fpath = os.path.join(save_path, fname)
            unzip.unzip_bz2(fpath, new_fpath)
                
            if 'BJX' in fname:
                zh, zdr, phidp, kdp, cc = read.metstar(new_fpath)
                # with open(new_fpath[:-4]+'.dat', 'wb') as new_file:
                #     new_file.write(struct.pack('f'*len(zh.flatten()), *zh.flatten()))
                #     new_file.write(struct.pack('f'*len(zdr.flatten()), *zdr.flatten()))
                #     new_file.write(struct.pack('f'*len(phidp.flatten()), *phidp.flatten()))
                #     new_file.write(struct.pack('f'*len(kdp.flatten()), *kdp.flatten()))
                #     new_file.write(struct.pack('f'*len(cc.flatten()), *cc.flatten()))
                
                
                band = 'x'
                reso = 0.075
                
                zh, zdr, phidp, kdp, cc = qc.qc_dual(zh, zdr, phidp, kdp, cc, band = band, reso = reso)
                with open(new_fpath[:-4]+'qc.dat', 'wb') as new_file:
                    new_file.write(struct.pack('f'*len(zh.flatten()), *zh.flatten()))
                    new_file.write(struct.pack('f'*len(zdr.flatten()), *zdr.flatten()))
                    new_file.write(struct.pack('f'*len(phidp.flatten()), *phidp.flatten()))
                    new_file.write(struct.pack('f'*len(kdp.flatten()), *kdp.flatten()))
                    new_file.write(struct.pack('f'*len(cc.flatten()), *cc.flatten()))
            
            os.remove(new_fpath)
           
            
            time2 = datetime.datetime.now()
            print(time2-time1)