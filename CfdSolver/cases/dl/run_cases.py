import sys
import os
from multiprocessing import get_context, current_process

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import time
import yaml
import copy
import re

from plasma_oscillation import run
from cfdsolver.cases import make_cases
from cfdsolver.utils import create_dir

def set_nested(data, value, *args):
    """ Function to set arguments with value in nested dictionnary """
    element = args[0]
    if len(args) == 1:
        data[element] = value 
    else:
        set_nested(data[element], value, *args[1:])

def params(cases, base_cfg, base_cn):
    """ Create the configuration files for each run and yield it to be read by
    each run function """
    for ncase, case in cases.items():
        # deepcopy is very important for recursive copy
        case_cfg = copy.deepcopy(base_cfg)
        for key, value in case.items():
            n_slash = key.count("/")
            nstr = n_slash + 1
            str_re = r'(\w*)/' * n_slash + r'(\w*)'
            re_keys = re.search(str_re, key)
            keys_tmp = []
            for i in range(nstr):
                keys_tmp.append(re_keys.group(i + 1))
            set_nested(case_cfg, value, *keys_tmp)
        case_cfg['plasma']['casename'] = f'{base_cn}case_{ncase:d}/'
        yield case_cfg

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Multiple cases run')
    args.add_argument('-c', '--config', required=True, type=str,
                      help='Config file path (default: None)')
    args.add_argument('-np', '--n_procs', default=1, type=int,
                        help='Number of procs')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    cases, base_cfg, base_cn = make_cases(cfg)

    with get_context('spawn').Pool(processes=args.n_procs) as p:
        p.map(run, params(cases, base_cfg, base_cn))