########################################################################################################################
#                                                                                                                      #
#                                                  Metrics functions                                                   #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 09.03.2020                                        #
#                                                                                                                      #
########################################################################################################################

import torch


def accuracy(output, target):
    """ Computes accuracy of the current epoch. """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def residual(output, target):
    """ Computes the residual of the current epoch. """
    with torch.no_grad():
        res = torch.sum(torch.abs(output - target))
    return res


def normed_residual(output, target):
    """ Computes the L2 norm of the residual of the currend epoch. """
    with torch.no_grad():
        res = torch.sum((output - target) ** 2)
    return res
