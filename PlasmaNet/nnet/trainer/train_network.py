########################################################################################################################
#                                                                                                                      #
#                                           PlasmaNet.nnet: train model                                                #
#                                                                                                                      #
#                         Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 10.03.2020                         #
#                                                                                                                      #
########################################################################################################################

import os
import argparse
import collections

import numpy as np
import torch
import yaml

from pathlib import Path

import PlasmaNet.nnet.data.data_loaders as module_data
import PlasmaNet.nnet.model.loss as module_loss
import PlasmaNet.nnet.model.metric as module_metric
import PlasmaNet.nnet.model as module_arch
from PlasmaNet.nnet.parse_config import ConfigParser
from PlasmaNet.nnet.trainer import Trainer

# Fix random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def run_train(config):
    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # If General photo, network will have two input channels!
    if 'PhotoLoss' in config['loss']['args']['loss_list']:
        config['arch']['args']['scales']['scale_0'][0][0] = 2

    # Build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # Initialize model weights
    if config['initializer'] != 'off':
        def init_weights(m):
            if type(m) == torch.nn.Conv2d:
                getattr(torch.nn.init, config['initializer']['type'])(m.weight, **config['initializer']['args'])
        model.apply(init_weights)

    # Get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    metrics = [getattr(module_metric, metric) for metric in config['metrics']]

    # Build optimizer, learning rate scheduler
    # Disable scheduler by commenting all lines with lr_scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    if config['lr_scheduler'] != 'off':
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()

def main():
    args = argparse.ArgumentParser(description='PlasmaNet.nnet')
    args.add_argument('-c', '--config', required=True, type=str,
                        help='Config file path (default: None)')
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg_dict = yaml.safe_load(yaml_stream)

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

if __name__ == '__main__':
    main()