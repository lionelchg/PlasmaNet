########################################################################################################################
#                                                                                                                      #
#                                                PlasmaNet: train model                                                #
#                                                                                                                      #
#                         Guillaume Bogopolsky, Lionel Cheng, Ekhi Ajuria, CERFACS, 10.03.2020                         #
#                                                                                                                      #
########################################################################################################################

import argparse
import collections

import numpy as np
import torch

import PlasmaNet.data.data_loaders as module_data
import PlasmaNet.model.loss as module_loss
import PlasmaNet.model.metric as module_metric
import PlasmaNet.model.multiscalenet as module_arch
#import PlasmaNet.model.unet as module_arch
from PlasmaNet.parse_config import ConfigParser
from PlasmaNet.trainer import Trainer


# Fix random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def main(config):
    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

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


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PlasmaNet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to checkpoint to resume (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom CLI options to modify configuration from default values given in yaml file
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['-bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
