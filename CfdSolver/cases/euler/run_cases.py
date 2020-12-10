import sys
import os
from multiprocessing import get_context, current_process

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import copy
import re
import time
import yaml

from plasma_oscillation import run
from cfdsolver.cases import make_cases
from cfdsolver.utils import create_dir

re_keys = re.compile(r'(\w*)/(\w*)')

def params(cases, base_cfg, base_cn):
    for ncase, case in cases.items():
        # deepcopy is very important for recursive copy
        case_cfg = copy.deepcopy(base_cfg)
        for key, value in case.items():
            keys_search = re_keys.search(key)
            group_key, inner_key = keys_search.group(1), keys_search.group(2)
            case_cfg[group_key][inner_key] = value
        case_cfg['casename'] = f'{base_cn}case_{ncase:d}/'
        yield case_cfg

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Multiple cases run')
    args.add_argument('-np', '--n_procs', default=None, type=int,
                        help='number of procs')
    args = args.parse_args()

    with open('run_cases.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    cases, base_cfg, base_cn = make_cases(cfg)

    print(cases)

    # # params(cases, base_cfg, base_cn)
    # with get_context('spawn').Pool(processes=args.np) as p:
    #     p.map(run, params(cases, base_cfg, base_cn))
