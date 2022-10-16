import os
import gzip


def unzip_one_file(input_file_path: str, output_file_path: str) -> None:
    with open(output_file_path, 'wb') as f_out:
        with gzip.open(input_file_path, 'rb') as f_in:
            f_out.write(f_in.read())


def unzip(source_root: str, target_root: str) -> None:
    if not os.path.exists(target_root):
        os.mkdir(target_root)
    date_list = sorted(os.listdir(source_root))
    for date in date_list:
        file_list = sorted(os.listdir(os.path.join(source_root, date)))
        if not os.path.exists(os.path.join(target_root, date)):
            os.mkdir(os.path.join(target_root, date))
        for file_ in file_list:
            source_file_path = os.path.join(source_root, date, file_)
            target_file_path = os.path.join(target_root, date, file_.replace('.gz', ''))
            unzip_one_file(source_file_path, target_file_path)
            print(target_file_path)


if __name__ == '__main__':
    source_root = './SBandBasic'
    target_root = './SBandBasicUnzip'
    unzip(source_root, target_root)
