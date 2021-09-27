########################################################################################################################
#                                                                                                                      #
#                                          Test the rotational operator                                                #
#                                                                                                                      #
#                               Lionel Cheng, Guillaume Bogopolsky, CERFACS, 02.04.2020                                #
#                                                                                                                      #
########################################################################################################################

import torch
import numpy as np
import pytest
from PlasmaNet.nnet.operators.rotational import scalar_rot as rot
from PlasmaNet.common.operators_torch import create_grid


def test_scalar_rotational():
    """ Test the scalar rotational operator on an analytical case. """
    # Create test grid
    nchannels, nx, ny, dx, dy, X, Y = create_grid()

    # Field and analytical solution initialisation
    field = torch.zeros((nchannels, 2, ny, nx)).type(torch.float64)
    analytical = torch.zeros((nchannels, 1, ny, nx)).type(torch.float64)
    for channel in range(nchannels):
        field[channel, 0, :, :] = - Y ** 2
        field[channel, 1, :, :] = X ** 2
        analytical[channel, 0, :, :] = 2 * X + 2 * Y

    # Compute rotational
    computed = rot(field, dx, dy)

    assert torch.allclose(computed, analytical)
    return X, Y, computed, analytical, field


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # scalar_rot

    x, y, computed, analytical, field = test_scalar_rotational()

    arrow_step = 5
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
    q = ax1.quiver(x[::arrow_step, ::arrow_step], y[::arrow_step, ::arrow_step],
                  field[0, 0, ::arrow_step, ::arrow_step], field[0, 1, ::arrow_step, ::arrow_step], pivot='mid')
    ax1.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

    p1 = ax2.contourf(x, y, computed[0, 0], 100)
    fig.colorbar(p1, label='Computed scalar rotational', ax=ax2)

    plt.savefig('test_scalar_rotational.png')
