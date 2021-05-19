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
import numpy as np

def residual(output, target, config):
    """ Computes the residual of the current batch. """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        res = torch.sum(torch.abs(output - target)) / config.nnx / config.nny / config.batch_size
    return res


def l2_norm(output, target, config):
    """ Computes the L2 norm of the residual of the current batch. """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        res = torch.sqrt(torch.sum((output - target) ** 2) / config.nnx / config.nny / config.batch_size)
    return res


def inf_norm(output, target, config):
    """ Computes the infinity norm of the residual of the current batch. """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        res = torch.max(torch.abs(output - target))
    return res

def Eresidual(output, target, config):
    """ Computes the electric field residual of the current batch. """
    elec_output = - gradient_scalar(output, config.dx, config.dy)
    elec_target = - gradient_scalar(target, config.dx, config.dy)
    with torch.no_grad():
        res = torch.sum(torch.abs(elec_output - elec_target)) \
                / config.nnx / config.nny / config.batch_size / 2
    return res


def El2_norm(output, target, config):
    """ Computes the electric field L2 norm of the residual of the current batch. """
    elec_output = - gradient_scalar(output, config.dx, config.dy)
    elec_target = - gradient_scalar(target, config.dx, config.dy)
    with torch.no_grad():
        res = torch.sqrt(torch.sum((elec_output - elec_target) ** 2) 
            / config.nnx / config.nny / config.batch_size / 2)
    return res


def Einf_norm(output, target, config):
    """ Computes the electric field infinity norm of the residual of the current batch. """
    elec_output = - gradient_scalar(output, config.dx, config.dy)
    elec_target = - gradient_scalar(target, config.dx, config.dy)
    with torch.no_grad():
        res = torch.max(torch.abs(elec_output - elec_target))
    return res


def ratio_avg(output, target, config):
    """ Computes the average of the ratio between the output and target of the current batch. """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        res = torch.mean(output / target)


def pearsonr(output, target, config):
    """ Computes the Pearson correlation coefficient between the output and target of the current batch. """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        out_mean, tar_mean = output.mean(), target.mean()
        out_centered, tar_centered = output - out_mean, target - tar_mean
        r_num = torch.sum(out_centered * tar_centered)
        r_den = torch.sqrt(torch.sum(out_centered ** 2) * torch.sum(tar_centered ** 2))
        return r_num / r_den


def pearsonr2(output, target, config):
    """ Computes the squared Pearson correlation coefficient between the output and target of the current batch. """
    return pearsonr(output, target, config) ** 2


def rotational(output, target, config):
    """ Computes the rotational of the electric field computed from the solution of the current batch. """
    output = output[:,0]
    target = target[:,0]
    with torch.no_grad():
        elec = gradient_scalar(output, config.dx, config.dy)
        res = torch.sum(scalar_rot(elec, config.dx, config.dy))
    return res

def phi11(output, target, config):
    """ Compute the first mode amplitude of the output and target """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        phi11_out = 4 / config.Lx / config.Ly * torch.sum(torch.sin(np.pi * config.X / config.Lx) * 
            torch.sin(np.pi * config.Y / config.Ly) * output * config.dx * config.dy)
        phi11_target = 4 / config.Lx / config.Ly * torch.sum(torch.sin(np.pi * config.X / config.Lx) * 
            torch.sin(np.pi * config.Y / config.Ly) * target * config.dx * config.dy)
        res = torch.abs(phi11_out - phi11_target) / config.batch_size
    return res
