import os
from multiprocessing import get_context

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import yaml

from plasma_oscillation import main
from cfdsolver.cases import make_cases

def params(cases, base_cfg, base_cn):
    for ncase, case in cases.items():
        print(case)
        for key, value in case.items():
            print(f'{key} {value}')
            base_cfg[key] = value
        base_cfg['casename'] = f'{base_cn}_{ncase:d}/'
        yield base_cfg

if __name__ == '__main__':

    with open('run_cases.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    cases, base_cfg, base_cn = make_cases(cfg)
    
    with get_context('spawn').Pool(processes=2) as p:
        p.imap(main, params(cases, base_cfg, base_cn))
