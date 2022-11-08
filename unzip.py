import os
import bz2

# fname = '20180716/SY/BJXSY.20180716.000000.AR2.bz2'
# # save_path = './'


# with open(fname[:-4], 'wb') as new_file, bz2.BZ2File(fname,  'rb') as file:
#     for data in iter(lambda : file.read(100 * 1024), b''):
#         new_file.write(data)
        


def unzip_bz2(path, save_path):
    ls = os.listdir(path)
    for filename in ls:
        if filename.endswith('bz2'):
            with bz2.BZ2File(os.path.join(path, filename), 'rb') as file, open(os.path.join(save_path, filename[:-4]), 'wb') as new_file:
                for data in iter(lambda : file.read(100 * 1024), b''):
                    new_file.write(data)
path = '20180716'
# save_path = os.path.join(path, 'raw')
# os.makedirs(save_path)    
unzip_bz2(path, path)