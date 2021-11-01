########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from numba import njit

lambda_j_three = np.array([0.0553, 0.1460,0.89]) * 1.0e2
A_j_three = np.array([1.986e-4, 0.0051, 0.4886]) * (1.0e2)**2
lambda_j_two = np.array([0.0974, 0.5877]) * 1.0e2
A_j_two = np.array([0.0021, 0.1775]) * (1.0e2)**2

coef_p = 0.038
quenching_press = 30 * 133

# Atmospheric Pressure in Torr
pO2 = 150

def photo_axisym(dx, dr, nx, nr, R, coeff, scale):
    diags = np.zeros((5, nx * nr))
    r = R.reshape(-1)

    # Filling the diagonals, first the down neumann bc,: the dirichlet bc and finally the interior nodes
    for i in range(nx * nr):
        if 0 < i < nx - 1:
            diags[0, i] = - (2 / dx**2 + 4 / dr**2 + coeff) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = 4 / dr**2 * scale
        elif i >= (nr - 1) * nx or i % nx == 0 or i % nx == nx - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, nx * nr - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + nx, nx * nr - 1)] = 0
            diags[4, max(i - nx, 0)] = 0
        else:
            diags[0, i] = - (2 / dx**2 + 2 / dr**2 + coeff) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = (1 + dr / (2 * r[i])) / dr**2 * scale
            diags[4, i - nx] = (1 - dr / (2 * r[i])) / dr**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * nr, nx * nr)))

@njit(cache=True)
def photo_coeff(E_p):
    # In Zheleznyak paper, the tabulation is done with E/p in V/cm * mmHg
    E_p = E_p * 133.32 / 100
    if E_p < 30:
        pcoeff = 5.e-2
    elif 30 <= E_p < 50:
        pcoeff = 0.07 / 20 * (E_p - 30) + 5.e-2
    elif 50 <= E_p < 100:
        pcoeff = 0.12 - 4e-2 / 50 * (E_p - 50)
    else:
        pcoeff = 0.08 - 2e-2 / 100 * (E_p - 100)
    return pcoeff

