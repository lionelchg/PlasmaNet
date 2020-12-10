import re
import copy
from itertools import product
from .utils import create_dir

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
            cases[index] = {}
            for i in range(len(list_keys)):
                cases[index][list_keys[i]] = element[i]

    elif mode == 'seq':
        for ncase in range(len(list_params[0])):
            cases[ncase] = {}
            for nkey in range(len(list_keys)):
                cases[ncase][list_keys[nkey]] = list_params[nkey][ncase]

    # Write in cases.log the difference cases and their description
    create_dir(base_cn)
    fp = open(base_cn + 'cases.log', 'w')
    fp.write(description + '\n\n')
    for ncase, case in cases.items():
        fp.write('------------------\n')
        fp.write(f'Case {ncase:d}\n')
        for key, value in case.items():
            fp.write(f'{key} = {value:.2e}\n')
    fp.close()

    return cases, base_cfg, base_cn

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