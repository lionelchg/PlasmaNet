########################################################################################################################
#                                                                                                                      #
#                                                  Metrics functions                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch


def residual(output, target, config):
    """ Computes the residual of the current batch. """
    with torch.no_grad():
        res = torch.sum(torch.abs(output - target) * config.voln) / config.surface
    return res


def l2_norm(output, target, config):
    """ Computes the L2 norm of the residual of the current batch. """
    with torch.no_grad():
        res = torch.sqrt(torch.sum((output - target) ** 2 * config.voln) / config.surface)
    return res


def inf_norm(output, target, config):
    """ Computes the infinity norm of the residual of the current batch. """
    with torch.no_grad():
        res = torch.max(torch.abs(output - target) * config.voln) / config.surface
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
