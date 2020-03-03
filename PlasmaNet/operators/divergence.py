########################################################################################################################
#                                                                                                                      #
#                                                 Divergence operator                                                  #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 28.02.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch


def divergence(field, dx, dy):
    """
    Calculates the divergence of a vector field (second order accurate, degraded to first order on boundaries).
    The output shape is the same as the input shape.

    Parameters
    ----------
    field : torch.Tensor
        Input 2D field: tensor of size (batch_size, 2, H, W)

    dx, dy : float

    Returns
    -------
    torch.Tensor
        Output divergence (scalar field)
    """

    # Create divergence tensor with shape (batch_size, h, w)
    divergence = torch.zeros_like(field[:, 0]).type(field.type()).unsqueeze(1)

    # Check sizes
    assert field.dim() == 4 and divergence.dim() == 4, 'Dimension mismatch'
    assert field.size(1) == 2, 'field is not 2D'

    assert field.is_contiguous() and divergence.is_contiguous(), 'Input is not contiguous'

    # Compute divergence (correction using linear interpolation at boundaries)
    # i <-> y and j <-> x, so field[:, 0, :, :] is field_x and field[:, 1, :, :] is field_y
    # center of the array
    divergence[:, 0, 1:-1, 1:-1] = (field[:, 0, 1:-1, 2:] - field[:, 0, 1:-1, :-2]) / (2 * dx) + \
                                   (field[:, 1, 2:, 1:-1] - field[:, 1, :-2, 1:-1]) / (2 * dy)

    # array sides except corners (respectively upper, lower, left and right sides)
    divergence[:, 0, 0, 1:-1] = (field[:, 0, 0, 2:] - field[:, 0, 0, :-2]) / (2 * dx) + \
                                (4 * field[:, 1, 1, 1:-1] - 3 * field[:, 1, 0, 1:-1] - field[:, 1, 2, 1:-1]) / (2 * dy)
    divergence[:, 0, -1, 1:-1] = (field[:, 0, -1, 2:] - field[:, 0, -1, :-2]) / (2 * dx) + \
                                 (3 * field[:, 1, -1, 1:-1] - 4 * field[:, 1, -2, 1:-1] + field[:, 1, -3, 1:-1]) / \
                                 (2 * dy)
    divergence[:, 0, 1:-1, 0] = (4 * field[:, 0, 1:-1, 1] - 3 * field[:, 0, 1:-1, 0] - field[:, 0, 1:-1, 2]) / \
                                (2 * dx) + (field[:, 1, 2:, 0] - field[:, 1, :-2, 0]) / (2 * dy)
    divergence[:, 0, 1:-1, -1] = (3 * field[:, 0, 1:-1, -1] - 4 * field[:, 0, 1:-1, -2] + field[:, 0, 1:-1, -3]) / \
                                 (2 * dx) + (field[:, 1, 2:, -1] - field[:, 1, :-2, -1]) / (2 * dy)

    # corners (respectively upper left, upper right, lower left and lower right)
    divergence[:, 0, 0, 0] = (4 * field[:, 0, 0, 1] - 3 * field[:, 0, 0, 0] - field[:, 0, 0, 2]) / (2 * dx) + \
                             (4 * field[:, 0, 1, 0] - 3 * field[:, 0, 0, 0] - field[:, 0, 2, 0]) / (2 * dy)
    divergence[:, 0, -1, 0] = (4 * field[:, 0, -1, 1] - 3 * field[:, 0, -1, 0] - field[:, 0, -1, 2]) / (2 * dx) + \
                              (3 * field[:, 1, -1, 0] - 4 * field[:, 1, -2, 0] + field[:, 1, -3, 0]) / (2 * dy)
    divergence[:, 0, 0, -1] = (3 * field[:, 0, 0, -1] - 4 * field[:, 0, 0, -2] + field[:, 0, 0, -3]) / (2 * dx) +\
                              (4 * field[:, 1, 1, -1] - 3 * field[:, 1, 0, -1] - field[:, 1, 2, -1]) / (2 * dy)
    divergence[:, 0, -1, -1] = (3 * field[:, 0, -1, -1] - 4 * field[:, 0, -1, -2] + field[:, 0, -1, -3]) / (2 * dx) + \
                               (3 * field[:, 1, -1, -1] - 4 * field[:, 1, -2, -1] + field[:, 1, -3, -1]) / (2 * dy)

    return divergence
