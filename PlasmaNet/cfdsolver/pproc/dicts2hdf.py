import glob
import pandas as pd
import argparse
from ...common.utils import load_obj


def dicts2hdf(data_dir, h5_fn, data_name):
    """ Converts dictionnaries in .pkl format to a single pandas 
    DataFrame stored in HDF5 format """
    files = glob.glob(data_dir + '**/*.pkl', recursive=True)

    data = dict()
    tmp_dict = load_obj(files[0])
    for key in tmp_dict.keys():
        data[key] = []

    for file in files:
        tmp_dict = load_obj(file)
        for key, value in tmp_dict.items():
            data[key].append(value)
    
    df = pd.DataFrame.from_dict(data)

    df.to_hdf(h5_fn, data_name)


if __name__ == '__main__':
    
    args = argparse.ArgumentParser(
                description='Conversion of dictionnaries to h5 in DataFrame')
    args.add_argument('-d', '--data_dir', required=True, type=str, help='Data directory')
    args.add_argument('-f', '--filename', required=True, type=str, help='Filename to write on')
    args.add_argument('-g', '--groupname', required=True, type=str, help='Groupname in HDF5 file')
    args = args.parse_args()

    dicts2hdf(args.data_dir, args.filename, args.groupname)
