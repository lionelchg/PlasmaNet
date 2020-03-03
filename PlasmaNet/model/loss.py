########################################################################################################################
#                                                                                                                      #
#                                                   Loss functions                                                     #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
from ..operators.laplacian import laplacian as lapl


def laplacian_loss(*args, **kwargs):
    """ A Laplacian loss function. """

    raise NotImplementedError('laplacian_loss is not implemented')
