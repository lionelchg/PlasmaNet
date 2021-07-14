########################################################################################################################
#                                                                                                                      #
#                                     PlasmaNet.nnet: train multiple models                                            #
#                                                                                                                      #
#                        Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 14.07.2021                          #
#                                                                                                                      #
########################################################################################################################

import os
import argparse
import re
import yaml
import copy
from pathlib import Path
from itertools import product

import numpy as np
import torch

from .train_network import run_train
from ..parse_config import ConfigParser

# Fix random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def generate_train_cfgs(cfg):
    """ Create networks dictionary from config file.
    Parameters different from base/mode/casename are used
    to create the number of cases. 'tree' mode enables a tree like structure
    create a different case for every parameter of the ranges provided whereas
    seq mode implies the same size for all parameter ranges """

    # base casename for data/fig directories
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

    networks = {}
    if mode == 'tree':
        product_params = list(product(*list_params))

        networks = {}
        for index, element in enumerate(product_params):
            networks[index + 1] = {}
            for i in range(len(list_keys)):
                networks[index + 1][list_keys[i]] = element[i]

    elif mode == 'seq':
        for ncase in range(len(list_params[0])):
            networks[ncase + 1] = {}
            for nkey in range(len(list_keys)):
                networks[ncase + 1][list_keys[nkey]] = list_params[nkey][ncase]

    return networks, base_cfg

def set_nested(data, value, *args):
    """ Function to set arguments with value in nested dictionary """
    element = args[0]
    if len(args) == 1:
        data[element] = value 
    else:
        set_nested(data[element], value, *args[1:])


def run_trains(networks, base_cfg):
    """ Create the configuration files for each run and yield it to be read by
    each run function """
    for ncase, network in networks.items():
        # deepcopy is very important for recursive copy
        case_cfg = copy.deepcopy(base_cfg)
        for key, value in network.items():
            n_slash = key.count("/")
            nstr = n_slash + 1
            str_re = r'(\w*)/' * n_slash + r'(\w*)'
            re_keys = re.search(str_re, key)
            keys_tmp = []
            for i in range(nstr):
                keys_tmp.append(re_keys.group(i + 1))
            set_nested(case_cfg, value, *keys_tmp)
        
        # Run training
        run(case_cfg)

def run(cfg_dict):
    """ Run training from dictionnary cfg_dict """
    # Architecture parsing in database
    if 'db_file' in cfg_dict['arch']:
        with open(Path(os.getenv('ARCHS_DIR')) / cfg_dict['arch']['db_file']) as yaml_stream:
            archs = yaml.safe_load(yaml_stream)
        tmp_cfg_arch = archs[cfg_dict['arch']['name']]
        if 'args' in cfg_dict['arch']:
            tmp_cfg_arch['args'] = {**cfg_dict['arch']['args'], **tmp_cfg_arch['args']}
        cfg_dict['arch'] = tmp_cfg_arch
    
    # Resume the training or not depending on entry in yaml
    if 'resume' in cfg_dict:
        resume = cfg_dict['resume']
    else:
        resume = None
    
    # Creation of config object
    config = ConfigParser(cfg_dict, resume)
    run_train(config)

def main():
    args = argparse.ArgumentParser(description='PlasmaNet.nnet')
    args.add_argument('-c', '--config', required=True, type=str,
                        help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg_networks = yaml.safe_load(yaml_stream)

    # Parse main config file
    networks, base_cfg = generate_train_cfgs(cfg_networks)

    # Run the different wanted configurations
    run_trains(networks, base_cfg)

if __name__ == '__main__':
    main()