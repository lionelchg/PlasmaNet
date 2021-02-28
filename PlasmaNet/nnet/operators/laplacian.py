########################################################################################################################
#                                                                                                                      #
#                                                 Laplacian operator                                                   #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 02.03.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch


def laplacian(field, dx, dy, b=0, r=None):
    """
    Calculates the laplacian of a scalar field (second order accurate, degraded to first order on boundaries).
    The output shape is the same as the input shape.

    Parameters
    ----------
    field : torch.Tensor
        Input 2D field: tensor of size (batch, 1, H, W)

    dx, dy : float

    b : float
        Parameter for the discretisation shape (see Hirsch p.164)

    Returns
    -------
    Tensor
        Output laplacian (scalar field)
    """

    # Create laplacian tensor with shape (batch_size, 1, h, w)
    laplacian = torch.zeros_like(field).type(field.type())

    # Check sizes
    assert field.dim() == 4 and laplacian.dim() == 4, 'Dimension mismatch'

    assert field.is_contiguous() and laplacian.is_contiguous(), 'Input is not contiguous'

    # Compute laplacian (correction using linear interpolation at boundaries)
    # center of the array
    laplacian[:, 0, 1:-1, 1:-1] = \
        (1 - b) * ((field[:, 0, 2:, 1:-1] + field[:, 0, :-2, 1:-1] - 2 * field[:, 0, 1:-1, 1:-1]) / dy**2 +
        (field[:, 0, 1:-1, 2:] + field[:, 0, 1:-1, :-2] - 2 * field[:, 0, 1:-1, 1:-1]) / dx**2) + \
        b * (field[:, 0, 2:, 2:] + field[:, 0, 2:, :-2] + field[:, 0, :-2, :-2] + field[:, 0, :-2, 2:] - 4 * field[:, 0, 1:-1, 1:-1]) \
        / (2 * dx**2)

    # array sides except corners (respectively upper, lower, left and right sides)
    # laplacian[:, 0, 0, 1:-1] = \
    #     (2 * field[:, 0, 0, 1:-1] - 5 * field[:, 0, 1, 1:-1] + 4 * field[:, 0, 2, 1:-1] - field[:, 0, 3, 1:-1]) / dy**2 + \
    #     (field[:, 0, 0, 2:] + field[:, 0, 0, :-2] - 2 * field[:, 0, 0, 1:-1]) / dx**2
    if r is None:
        laplacian[:, 0, 0, 1:-1] = \
            (2 * field[:, 0, 0, 1:-1] - 5 * field[:, 0, 1, 1:-1] + 4 * field[:, 0, 2, 1:-1] - field[:, 0, 3, 1:-1]) / dy**2 + \
            (field[:, 0, 0, 2:] + field[:, 0, 0, :-2] - 2 * field[:, 0, 0, 1:-1]) / dx**2
    else:
        laplacian[:, 0, 0, 1:-1] = (field[:, 0, 0, 2:] + field[:, 0, 0, :-2] - 2 * field[:, 0, 0, 1:-1]) / dx**2
    laplacian[:, 0, -1, 1:-1] = \
        (2 * field[:, 0, -1, 1:-1] - 5 * field[:, 0, -2, 1:-1] + 4 * field[:, 0, -3, 1:-1] - field[:, 0, -4, 1:-1]) / dy**2 + \
        (field[:, 0, -1, 2:] + field[:, 0, -1, :-2] - 2 * field[:, 0, -1, 1:-1]) / dx**2
    laplacian[:, 0, 1:-1, 0] = \
        (field[:, 0, 2:, 0] + field[:, 0, :-2, 0] - 2 * field[:, 0, 1:-1, 0]) / dy**2 + \
        (2 * field[:, 0, 1:-1, 0] - 5 * field[:, 0, 1:-1, 1] + 4 * field[:, 0, 1:-1, 2] - field[:, 0, 1:-1, 3]) / dx**2
    laplacian[:, 0, 1:-1, -1] = \
        (field[:, 0, 2:, -1] + field[:, 0, :-2, -1] - 2 * field[:, 0, 1:-1, -1]) / dy**2 + \
        (2 * field[:, 0, 1:-1, -1] - 5 * field[:, 0, 1:-1, -2] + 4 * field[:, 0, 1:-1, -3] - field[:, 0, 1:-1, -4]) / dx**2

    # # corners (respectively upper left, upper right, lower left and lower right)
    # laplacian[:, 0, 0, 0] = \
    #     (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 1, 0] + 4 * field[:, 0, 2, 0] - field[:, 0, 3, 0]) / dy**2 + \
    #     (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 0, 1] + 4 * field[:, 0, 0, 2] - field[:, 0, 0, 3]) / dx**2
    # laplacian[:, 0, 0, -1] = \
    #     (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 1, -1] + 4 * field[:, 0, 2, -1] - field[:, 0, 3, -1]) / dy**2 + \
    #     (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2

    if r is None:
        laplacian[:, 0, 0, 0] = \
            (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 1, 0] + 4 * field[:, 0, 2, 0] - field[:, 0, 3, 0]) / dy**2 + \
            (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 0, 1] + 4 * field[:, 0, 0, 2] - field[:, 0, 0, 3]) / dx**2
        laplacian[:, 0, 0, -1] = \
            (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 1, -1] + 4 * field[:, 0, 2, -1] - field[:, 0, 3, -1]) / dy**2 + \
            (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2
    else:
        laplacian[:, 0, 0, 0] = (2 * field[:, 0, 0, 0] - 5 * field[:, 0, 0, 1] + 4 * field[:, 0, 0, 2] - field[:, 0, 0, 3]) / dx**2
        laplacian[:, 0, 0, -1] = (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2

    laplacian[:, 0, -1, 0] = \
        (2 * field[:, 0, -1, 0] - 5 * field[:, 0, -2, 0] + 4 * field[:, 0, -3, 0] - field[:, 0, -4, 0]) / dy**2 + \
        (2 * field[:, 0, -1, 0] - 5 * field[:, 0, -1, 1] + 4 * field[:, 0, -1, 2] - field[:, 0, -1, 3]) / dx**2
    laplacian[:, 0, -1, -1] = \
        (2 * field[:, 0, -1, -1] - 5 * field[:, 0, -2, -1] + 4 * field[:, 0, -3, -1] - field[:, 0, -4, -1]) / dy**2 + \
        (2 * field[:, 0, 0, -1] - 5 * field[:, 0, 0, -2] + 4 * field[:, 0, 0, -3] - field[:, 0, 0, -4]) / dx**2

    if r is not None:
        laplacian[:, 0, 1:-1, :] += (field[:, 0, 2:, :] - field[:, 0, :-2, :]) / (2 * dy) / r[1:-1, :]
        laplacian[:, 0, 0, :] += 4 * (field[:, 0, 1, :] - field[:, 0, 0, :]) / dy**2
        laplacian[:, 0, -1, :] += (3 * field[:, 0, -1, :] - 4 * field[:, 0, -2, :] + field[:, 0, -3, :]) / (2 * dy) / r[-1, :]

    return laplacian
