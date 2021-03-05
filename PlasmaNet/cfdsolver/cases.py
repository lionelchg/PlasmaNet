import re
import copy
from itertools import product
import yaml
import argparse
from multiprocessing import get_context, current_process

# Local imports
from .utils import create_dir
from .scalar.scalar import ScalarTransport
from .scalar.streamer import StreamerMorrow
from .euler.euler import Euler
from .euler.plasma import PlasmaEuler

def make_cases(cfg):
    """ Create cases dictionnary from config file.
    Parameters different from base/mode/casename are used
    to create the number of cases. 'tree' mode enables a tree like structure
    create a different case for every parameter of the ranges provided whereas
    seq mode implies the same size for all parameter ranges """

    # base casename for data/fig directories
    base_cn = cfg['casename']
    del cfg['casename']
    description = cfg['description']
    del cfg['description']

    # Base config that will be modified by the parameters
    base_cfg = cfg['base']
    del cfg['base']

    # tree/seq mode depending on the way parameters are expanded
    mode = cfg['mode']
    del cfg['mode']

    # creation of list of keys and params
    list_keys, list_params = [], []
    for key, value in cfg.items():
        list_keys.append(key)
        list_params.append(value)
    

    cases = {}
    if mode == 'tree':
        product_params = list(product(*list_params))

        cases = {}
        for index, element in enumerate(product_params):
            cases[index + 1] = {}
            for i in range(len(list_keys)):
                cases[index + 1][list_keys[i]] = element[i]

    elif mode == 'seq':
        for ncase in range(len(list_params[0])):
            cases[ncase + 1] = {}
            for nkey in range(len(list_keys)):
                cases[ncase + 1][list_keys[nkey]] = list_params[nkey][ncase]

    # Write in cases.log the difference cases and their description
    create_dir(base_cn)
    fp = open(base_cn + 'cases.log', 'w')
    fp.write(description + '\n\n')
    for ncase, case in cases.items():
        fp.write('------------------\n')
        fp.write(f'Case {ncase:d}\n')
        for key, value in case.items():
            if isinstance(value, float):
                fp.write(f'{key} = {value:.2e}\n')
            elif isinstance(value, int):
                fp.write(f'{key} = {value:d}\n')
            elif isinstance(value, list):
                fp.write(key + ' = ' + str(value) + '\n')
            elif isinstance(value, str):
                fp.write(f'{key} = {value}\n')

    fp.close()

    return cases, base_cfg, base_cn

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
        case_cfg['casename'] = f'{base_cn}case_{ncase:d}/'
        yield case_cfg

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Multiple cases run')
    args.add_argument('-np', '--n_procs', default=None, type=int,
                        help='Number of procs')
    args.add_argument('-c', '--config', type=str,
                        help='Config filename', required=True)
    args.add_argument('-t', '--type', type=str,
                        help='Type of simulation: scalar/streamer/euler/pleuler', required=True)
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    
    cases, base_cfg, base_cn = make_cases(cfg)

    with get_context('spawn').Pool(processes=args.n_procs) as p:
        if args.type == 'scalar':
            p.map(ScalarTransport.run, params(cases, base_cfg, base_cn))
        elif args.type == 'streamer':
            p.map(StreamerMorrow.run, params(cases, base_cfg, base_cn))
        elif args.type == 'euler':
            p.map(Euler.run, params(cases, base_cfg, base_cn))
        elif args.type == 'pleuler':
            p.map(PlasmaEuler.run, params(cases, base_cfg, base_cn))
    