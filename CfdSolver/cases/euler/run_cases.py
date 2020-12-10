import sys
import os
from multiprocessing import get_context, current_process

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import time
import yaml

from plasma_oscillation import run
from cfdsolver.cases import make_cases, params
from cfdsolver.utils import create_dir

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Multiple cases run')
    args.add_argument('-np', '--n_procs', default=None, type=int,
                        help='number of procs')
    args = args.parse_args()

    with open('run_cases.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    cases, base_cfg, base_cn = make_cases(cfg)

    with get_context('spawn').Pool(processes=args.np) as p:
        p.map(run, params(cases, base_cfg, base_cn))