########################################################################################################################
#                                                                                                                      #
#                                                  Gradient operator                                                   #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 02.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np


def gradient_numpy(field, dx, dy):
    """
    Computes the gradient of a scalar field using the numpy.gradient function.

    Parameters
    ----------
    field : torch.Tensor
        Input scalar field: tensor of size (batch_size, 1, H, W)

    dx, dy : float

    Returns
    -------
    torch.Tensor
        Output gradient field: tensor of size (batch_size, 2, H, W)
    """

    # Create gradient tensor with shape (batch_size, 2, H, W)
    batch_size, h, w = field.size(0), field.size(2), field.size(3)
    gradient = torch.zeros((batch_size, 2, h, w)).type(field.type())

    # Check sizes
    assert field.dim() == 4 and gradient.dim() == 4, 'Dimension mismatch'
    assert field.size(1) == 1, 'field is not scalar'

    assert field.is_contiguous() and gradient.is_contiguous(), 'Input is not contiguous'

    # Compute the gradient
    for i in range(batch_size):
        tmp0, tmp1 = np.gradient(field[i, 0], dy, dx)
        gradient[i, 0] = torch.from_numpy(tmp0.T)
        gradient[i, 1] = torch.from_numpy(tmp1.T)

    return gradient



