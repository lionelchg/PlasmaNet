########################################################################################################################
#                                                                                                                      #
#                                     Function to compute the first Fourier mode                                       #
#                                   as an initial guess for the Dirichlet Network                                      #
#                                                                                                                      #
#                                         Ekhi Ajuria, CERFACS, 29.04.2020                                             #
#                                                                                                                      #
########################################################################################################################

from itertools import repeat
from pathlib import Path
from .dirichlet import sum_series
import pandas as pd
import yaml
import numpy as np
import torch

def fourier_guess(BC, modes):
    """ Give a guess for the 2D network

    Args:
        BC (array[bsz, 1, resY]): Potential on the BC for Dirichlet Network
        modes (int): Number of modes to take into account in order to make the gues

    Returns:
        [array[bsz, 1, resY, resX]]: Guessed potential
    """
    assert BC.size(1) == 1, "Size for array not OK, dim should be 1, not {}".format(BC.size(1))
    bsz  = BC.size(0)
    resY = BC.size(2)
    resX = resY

    # First convert into a numpy array to make the guess
    BC_np = BC.numpy()

    n_points = resY
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin

    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    X_ex = np.repeat(X[ np.newaxis, : , :], bsz, axis=0)
    Y_ex = np.repeat(Y[ np.newaxis, : , :], bsz, axis=0)

    # Call the function that calculates the guess
    pot_th = sum_series(BC_np[:,0], X_ex, Y_ex, Lx, Ly, modes)
 
    #Convert to tprch and unsqueeze to the good shape
    guess = torch.from_numpy(pot_th).unsqueeze(1).expand((bsz, 1, resY, resX))

    # 3 modifications: First switch X and Y axes (numpy vs torch), then invert
    # X and Y so that the output has the correct shape. These modifications are
    # empirical, there's probably a better way to do ti
    final_guess = torch.flip(torch.flip(torch.transpose(guess, 3, 2),[3,2]),[1,2])

    return final_guess


