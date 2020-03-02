########################################################################################################################
#                                                                                                                      #
#                                           Test the divergence operator                                               #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 28.02.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch
import numpy as np
import pytest
from PlasmaNet.Operators.divergence import divergence as div


def test_divergence_2d():
    """ Test the divergence operator on an analytical case."""
    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    nx, ny = 101, 101
    nchannels = 10
    dx, dy = (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1)
    x, y = torch.linspace(xmin, xmax, nx), torch.linspace(ymin, ymax, ny)
    Y, X = torch.meshgrid(y, x)  # Pay attention to the reversed order of the axes with torch.Tensor !

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X**2
        field[channel, 1, :, :] = Y**2
        analytical[channel, 0, :, :] = 2 * X + 2 * Y

    # Compute divergence
    computed = div(field, dx, dy)

    # assert torch.all(computed.eq(analytical))
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    x, y, computed, analytical, field = test_divergence_2d()

    print((computed[0, 0, :, :] - analytical[0, 0, :, :]))
    print(computed[0, 0, :, :])
    print(analytical[0, 0, :, :])
    print(torch.allclose(computed[0], analytical[0]))

    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()
    p1 = ax1.contourf(x, y, analytical[0, 0, :, :], 100)
    cbar1 = fig.colorbar(p1, label='Analytical divergence field', ax=ax1)
    p2 = ax2.contourf(x, y, computed[0, 0, :, :], 100)
    cbar2 = fig.colorbar(p2, label='Computed divergence field', ax=ax2)
    p3 = ax3.contourf(x, y, torch.abs(computed[0, 0, :, :] - analytical[0, 0, :, :]) / analytical[0, 0, :, :], 100,
                      locator=ticker.LogLocator())
    cbar3 = fig.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_divergence_2d')
