########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import integrate

def coef(V_u, n, x, Lx):
    return integrate.simps(V_u * np.sin(n * np.pi * x / Lx), x)

def series_term(V_u, x, y, Lx, Ly, n):
    coe = coef(V_u, n, x[:,0, :], Lx)[ :, np.newaxis, np.newaxis]
    return coe* np.sin(n * np.pi * x / Lx) * np.sinh(n * np.pi * y / Lx) / np.sinh(n * np.pi * Ly / Lx)

def sum_series(V_u, x, y, Lx, Ly, N):
    series = np.zeros_like(x)
    for n in range(1, N + 1):
        series += series_term(V_u, x, y, Lx, Ly, n)
    return 2 / Lx * series