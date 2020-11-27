def make_cases(cfg):
    base_cfg = cfg['base']
    del cfg['base']

    mode = cfg['mode']
    del cfg['mode']

    base_cn = cfg['casename']
    del cfg['casename']

    list_keys, list_params = [], []
    for key, value in cfg.items():
        list_keys.append(key)
        list_params.append(value)

    cases = {}
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

    return cases, base_cfg, base_cn