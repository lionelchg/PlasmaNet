########################################################################################################################
#                                                                                                                      #
#                                         Helper functions for the tests                                               #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch


def create_grid():
    """
    Initializes a square cartesian mesh for the operators tests.

    Returns
    -------
    nchannels : int
        Number of channels

    nx, ny : int
        Number of elements

    dx, dy : float
        Step size

    X, Y : torch.Tensor
        Tensor containing the cartesian coordinates of size (ny, nx)
    """

    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    nx, ny = 101, 101
    nchannels = 10
    dx, dy = (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1)
    x, y = torch.linspace(xmin, xmax, nx), torch.linspace(ymin, ymax, ny)
    Y, X = torch.meshgrid(y, x)  # Pay attention to the reversed order of the axes with torch.Tensor !

    return nchannels, nx, ny, dx, dy, X, Y
