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
        ncase = 0
        if len(list_params) == 1:
            key = list_keys[0]
            params = list_params[0]
            for param in params:
                cases[ncase] = {}
                cases[ncase][key] = param
                ncase += 1
        elif len(list_params) == 2:
            key, key1 = list_keys[:2]
            params, params1 = list_params[:2]
            for param in params:
                for param1 in params1:
                    cases[ncase] = {}
                    cases[ncase][key] = param
                    cases[ncase][key1] = param1
                    ncase += 1
        elif len(list_params) == 3:
            key, key1, key2 = list_keys[:3]
            params, params1, params2 = list_params[:3]
            for param in params:
                for param1 in params1:
                    for param2 in params2:
                        cases[ncase] = {}
                        cases[ncase][key] = param
                        cases[ncase][key1] = param1
                        cases[ncase][key2] = param2
                        ncase += 1

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
