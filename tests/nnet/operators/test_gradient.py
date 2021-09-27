########################################################################################################################
#                                                                                                                      #
#                                           Test the gradient operators                                                #
#                                                                                                                      #
#                               Guillaume Bogopolsky, Lionel Cheng, CERFACS, 02.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import pytest
from PlasmaNet.nnet.operators.gradient import gradient_numpy, gradient_scalar, gradient_diag
from PlasmaNet.common.operators_torch import create_grid


def init_scalar_gradient():
    """ Return a scalar field and its analytical gradient, as well as the coordinates used. """
    # Create test grid
    nchannels, nx, ny, dx, dy, X, Y = create_grid()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X ** 2 + Y ** 2
        analytical[channel, 0, :, :] = 2 * X
        analytical[channel, 1, :, :] = 2 * Y

    return X, Y, dx, dy, field, analytical


def test_gradient_numpy():
    """ Test the gradient_numpy operator on an analytical case. """
    # Initialize
    X, Y, dx, dy, field, analytical = init_scalar_gradient()

    # Compute gradient
    computed = gradient_numpy(field, dx, dy)

    assert torch.allclose(computed, analytical, atol=.01001)  # atol to account for order degradation at the corners
    return X, Y, computed, analytical, field


def test_gradient_scalar():
    """ Test the gradient_scalar operator on an analytical case. """
    # Initialize
    X, Y, dx, dy, field, analytical = init_scalar_gradient()

    # Compute gradient
    computed = gradient_scalar(field, dx, dy)

    assert torch.allclose(computed, analytical, atol=1e-4)  # Better treatment of the boundaries
    return X, Y, computed, analytical, field


def test_gradient_diag():
    """
    Test the gradient_diag operator on an analytical case. The sum of the diagonal terms should be equal to the
    divergence of the input vector field.
    """
    # Create test grid
    nchannels, nx, ny, dx, dy, X, Y = create_grid()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = X ** 2
        field[channel, 1, :, :] = Y ** 2
        analytical[channel, 0, :, :] = 2 * X
        analytical[channel, 1, :, :] = 2 * Y

    # Compute gradient
    computed = gradient_diag(field, dx, dy)

    assert torch.allclose(computed, analytical)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # gradient_numpy

    x, y, computed, analytical, field = test_gradient_numpy()
    analytical_norm = torch.sqrt((analytical[0]**2).sum(0))
    computed_norm = torch.sqrt((computed[0]**2).sum(0))

    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()
    p1 = ax1.contourf(x, y, analytical_norm, 100)
    cbar1 = fig.colorbar(p1, label='Analytical gradient norm', ax=ax1)
    p2 = ax2.contourf(x, y, computed_norm, 100)
    cbar2 = fig.colorbar(p2, label='Computed gradient norm', ax=ax2)
    p3 = ax3.contourf(x, y, torch.abs(computed_norm - analytical_norm) / analytical_norm, 100,
                      locator=ticker.LogLocator())
    cbar3 = fig.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_gradient_numpy.png')

    # gradient_scalar

    x, y, computed, analytical, field = test_gradient_scalar()
    analytical_norm = torch.sqrt((analytical[0] ** 2).sum(0))
    computed_norm = torch.sqrt((computed[0] ** 2).sum(0))

    fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
    ax1, ax2, ax3, _ = axarr.ravel()
    p1 = ax1.contourf(x, y, analytical_norm, 100)
    cbar1 = fig.colorbar(p1, label='Analytical gradient norm', ax=ax1)
    p2 = ax2.contourf(x, y, computed_norm, 100)
    cbar2 = fig.colorbar(p2, label='Computed gradient norm', ax=ax2)
    p3 = ax3.contourf(x, y, torch.abs(computed_norm - analytical_norm) / analytical_norm, 100,
                      locator=ticker.LogLocator())
    cbar3 = fig.colorbar(p3, label='Relative difference', ax=ax3)
    plt.tight_layout()
    plt.savefig('test_gradient_scalar.png')

