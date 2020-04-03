########################################################################################################################
#                                                                                                                      #
#                                                  Metrics functions                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch


def residual(output, target, config):
    """ Computes the residual of the current epoch. """
    batch_size = config['data_loader']['args']['batch_size']
    n_x = config['globals']['size']
    with torch.no_grad():
        res = torch.sum(torch.abs(output - target)) / n_x**2 / batch_size
    return res


def l2_norm(output, target, config):
    """ Computes the L2 norm of the residual of the current epoch. """
    batch_size = config['data_loader']['args']['batch_size']
    n_x = config['globals']['size']
    with torch.no_grad():
        res = torch.sqrt(torch.sum((output - target) ** 2) / n_x**2 / batch_size)
    return res


def inf_norm(output, target, config):
    """ Computes the infinity norm of the residual of the current epoch. """
    with torch.no_grad():
        res = torch.max(torch.abs(output - target))
    return res
