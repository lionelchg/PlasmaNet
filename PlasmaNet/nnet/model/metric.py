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
        res = torch.sum(torch.abs(output - target)) / np.prod(output.shape[-2:]) / config.batch_size
    return res


def l2_norm(output, target, config):
    """ Computes the L2 norm of the residual of the current batch. """
    output = output[:, 0]
    target = target[:, 0]
    with torch.no_grad():
        res = torch.sqrt(torch.sum((output - target) ** 2) / np.prod(output.shape[-2:]) / config.batch_size)
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
                / np.prod(output.shape[-2:]) / config.batch_size / 2
    return res


def El2_norm(output, target, config):
    """ Computes the electric field L2 norm of the residual of the current batch. """
    elec_output = - gradient_scalar(output, config.dx, config.dy)
    elec_target = - gradient_scalar(target, config.dx, config.dy)
    with torch.no_grad():
        res = torch.sqrt(torch.sum((elec_output - elec_target) ** 2) 
            / np.prod(output.shape[-2:]) / config.batch_size / 2)
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
    """ Compute the (1, 1) mode amplitude of the output and target """
    output = output[:, 0]
    target = target[:, 0]
    xmin = config['globals']['xmin']
    xmax = config['globals']['xmax']
    x_line, y_line = np.linspace(xmin, xmax, output.size(1)), np.linspace(xmin, xmax, output.size(2))
    X, Y = np.meshgrid(x_line, y_line)
    with torch.no_grad():
        phi11_out = 4 / config.Lx / config.Ly * torch.sum(torch.sin(np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * output * config.dx * config.dy, dim=(1, 2))
        phi11_target = 4 / config.Lx / config.Ly * torch.sum(torch.sin(np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * target * config.dx * config.dy, dim=(1, 2))
        res = torch.sum(torch.abs(phi11_out - phi11_target)) / config.batch_size
    return res

def phi21(output, target, config):
    """ Compute the (2, 1) mode amplitude of the output and target """
    output = output[:, 0]
    target = target[:, 0]
    xmin = config['globals']['xmin']
    xmax = config['globals']['xmax']
    x_line, y_line = np.linspace(xmin, xmax, output.size(1)), np.linspace(xmin, xmax, output.size(2))
    X, Y = np.meshgrid(x_line, y_line)
    with torch.no_grad():
        phi11_out = 4 / config.Lx / config.Ly * torch.sum(torch.sin(2 * np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * output * config.dx * config.dy, dim=(1, 2))
        phi11_target = 4 / config.Lx / config.Ly * torch.sum(torch.sin(2 * np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * target * config.dx * config.dy, dim=(1, 2))
        res = torch.sum(torch.abs(phi11_out - phi11_target)) / config.batch_size
    return res

def phi12(output, target, config):
    """ Compute the (1, 2) mode amplitude of the output and target """
    output = output[:, 0]
    target = target[:, 0]
    xmin = config['globals']['xmin']
    xmax = config['globals']['xmax']
    x_line, y_line = np.linspace(xmin, xmax, output.size(1)), np.linspace(xmin, xmax, output.size(2))
    X, Y = np.meshgrid(x_line, y_line)
    with torch.no_grad():
        phi11_out = 4 / config.Lx / config.Ly * torch.sum(torch.sin(np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(2 * np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * output * config.dx * config.dy, dim=(1, 2))
        phi11_target = 4 / config.Lx / config.Ly * torch.sum(torch.sin(np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(2 * np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * target * config.dx * config.dy, dim=(1, 2))
        res = torch.sum(torch.abs(phi11_out - phi11_target)) / config.batch_size
    return res

def phi22(output, target, config):
    """ Compute the (2, 2) mode amplitude of the output and target """
    output = output[:, 0]
    target = target[:, 0]
    xmin = config['globals']['xmin']
    xmax = config['globals']['xmax']
    x_line, y_line = np.linspace(xmin, xmax, output.size(1)), np.linspace(xmin, xmax, output.size(2))
    X, Y = np.meshgrid(x_line, y_line)
    with torch.no_grad():
        phi22_out = 4 / config.Lx / config.Ly * torch.sum(torch.sin(2 * np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(2 * np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * output * config.dx * config.dy, dim=(1, 2))
        phi22_target = 4 / config.Lx / config.Ly * torch.sum(torch.sin(2 * np.pi * torch.from_numpy(X).cuda() / config.Lx) * 
            torch.sin(2 * np.pi * torch.from_numpy(Y).cuda() / config.Ly) 
            * target * config.dx * config.dy, dim=(1, 2))
        res = torch.sum(torch.abs(phi22_out - phi22_target)) / config.batch_size
    return res

