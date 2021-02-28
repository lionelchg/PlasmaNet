#                                                                                                                      #
#                                                 Rotational operator                                                  #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 02.04.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch


def scalar_rot(field, dx, dy):
    """
    Calculates the scalar rotational of a 2D vector field (second order accurate).
    The output shape is the same as the input shape.

    Parameters
    ----------
    field : torch.Tensor
        Input 2D field: tensor of size (batch_size, 2, H, W)

    dx, dy : float

    Returns
    -------
    torch.Tensor
        Output rotational (scalar field) 
    """

    # Create rotational tensor with shape (batch_size, 1, h, w)
    rotational = torch.zeros_like(field[:, 0]).type(field.type()).unsqueeze(1)

    # Check sizes
    assert field.dim() == 4 and rotational.dim() == 4, 'Dimension mismatch'
    assert field.size(1) == 2, 'field is not 2D'

    assert field.is_contiguous() and rotational.is_contiguous(), 'Input is not contiguous'

    # first compute dfield_y / dx
    rotational[:, 0, :, 1:-1] = (field[:, 1, :, 2:] - field[:, 1, :, :-2]) / (2 * dx)
    rotational[:, 0, :, 0] = (4 * field[:, 1, :, 1] - 3 * field[:, 1, :, 0] - field[:, 1, :, 2]) / (2 * dx)
    rotational[:, 0, :, -1] = (3 * field[:, 1, :, -1] - 4 * field[:, 1, :, -2] + field[:, 1, :, -3]) / (2 * dx)

    # second compute dfield_x / dy
    rotational[:, 0, 1:-1, :] -= (field[:, 0, 2:, :] - field[:, 0, :-2, :]) / (2 * dy)
    rotational[:, 0, 0, :] -= (4 * field[:, 0, 1, :] - 3 * field[:, 0, 0, :] - field[:, 0, 2, :]) / (2 * dy)
    rotational[:, 0, -1, :] -= (3 * field[:, 0, -1, :] - 4 * field[:, 0, -2, :] + field[:, 0, -3, :]) / (2 * dy)

    return rotational
