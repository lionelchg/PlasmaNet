########################################################################################################################
#                                                                                                                      #
#                                                  Metrics functions                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch
from ..operators.rotational import scalar_rot
from ..operators.gradient import gradient_scalar


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
        res = torch.sqrt(torch.sum((output - target) ** 2)) / n_x**2 / batch_size
    return res


def inf_norm(output, target, config):
    """ Computes the infinity norm of the residual of the current batch. """
    with torch.no_grad():
        res = torch.max(torch.abs(output - target)) * config.ds / config.surface
    return res


def ratio_avg(output, target, config):
    """ Computes the average of the ratio between the output and target of the current batch. """
    with torch.no_grad():
        res = torch.mean(output / target)


def pearsonr(output, target, config):
    """ Computes the Pearson correlation coefficient between the output and target of the current batch. """
    with torch.no_grad():
        out_mean, tar_mean = output.mean(), target.mean()
        out_centered, tar_centered = output - out_mean, target - tar_mean
        r_num = torch.sum(out_centered * tar_centered)
        r_den = torch.sqrt(torch.sum(out_centered ** 2) * torch.sum(tar_centered ** 2))
        return r_num / r_den


def rotational(output, target, config):
    """ Computes the rotational of the electric field computed from the solution of the current batch. """
    with torch.no_grad():
        elec = gradient_scalar(output, config.dx, config.dy)
        res = torch.sum(scalar_rot(elec, config.dx, config.dy))
    return res
