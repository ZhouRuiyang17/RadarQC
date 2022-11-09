import bz2

def unzip_bz2(fpath, new_fpath):
    with bz2.BZ2File(fpath, 'rb') as file, open(new_fpath, 'wb') as new_file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)