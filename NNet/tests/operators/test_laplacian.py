########################################################################################################################
#                                                                                                                      #
#                                           Test the laplacian operator                                                #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 28.02.2020                                #
#                                                                                                                      #
########################################################################################################################


import torch
import numpy as np
import pytest
from PlasmaNet.nnet.operators.laplacian import laplacian as lapl
from tests.operators import create_grid


def test_laplacian_2d():
    """ Test the laplacian operator on an analytical case."""
    # Create test grid
    nchannels, nx, ny, dx, dy, X, Y = create_grid()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        # field[channel, 0, :, :] = X**3 + Y**3
        # analytical[channel, 0, :, :] = 6 * X + 6 * Y
        field[channel, 0, :, :] = torch.exp(X) + torch.exp(2 * Y)
        analytical[channel, 0, :, :] = torch.exp(X) + 4 * torch.exp(2 * Y)

    # Compute laplacian
    computed = lapl(field, dx, dy, b=1)
    computed_1 = lapl(field, dx, dy, b=1 / 3)
    computed_2 = lapl(field, dx, dy, b=0)
    print('b = 1 : ', torch.sum(abs(computed[0, 0, :, :] - analytical[0, 0, :, :])))
    print('b = 1 / 3 : ', torch.sum(abs(computed_1[0, 0, :, :] - analytical[0, 0, :, :])))
    print('b = 0 : ', torch.sum(abs(computed_2[0, 0, :, :] - analytical[0, 0, :, :])))

    assert torch.allclose(computed, analytical, rtol=1e-2)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    x, y, computed, analytical, field = test_laplacian_2d()

    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()
    p1 = ax1.contourf(x, y, analytical[0, 0, :, :], 100)
    cbar1 = fig.colorbar(p1, label='Analytical laplacian field', ax=ax1)
    p2 = ax2.contourf(x, y, computed[0, 0, :, :], 100)
    cbar2 = fig.colorbar(p2, label='Computed laplacian field', ax=ax2)
    p3 = ax3.contourf(x, y, torch.abs(computed[0, 0, :, :] - analytical[0, 0, :, :]) / analytical[0, 0, :, :], 100,
                      locator=ticker.LogLocator())
    cbar3 = fig.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_laplacian_2d.png')
