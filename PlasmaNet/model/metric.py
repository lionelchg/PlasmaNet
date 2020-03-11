########################################################################################################################
#                                                                                                                      #
#                                                  Metrics functions                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch


def residual(output, target):
    """ Computes the residual of the current epoch. """
    with torch.no_grad():
        res = torch.sum(torch.abs(output - target))
    return res


def l2_norm(output, target):
    """ Computes the L2 norm of the residual of the currend epoch. """
    with torch.no_grad():
        res = torch.sum((output - target) ** 2)
    return res
