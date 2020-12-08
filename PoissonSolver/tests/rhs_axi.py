########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
import copy

from poissonsolver.poisson import Poisson
from poissonsolver.funcs import gaussian

fig_dir = 'figures/rhs_2D_axi/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

if __name__ == '__main__':
    xmin, xmax, nnx = 0, 4e-3, 201
    rmin, rmax, nnr = 0, 1e-3, 101
    x, r = np.linspace(xmin, xmax, nnx), np.linspace(rmin, rmax, nnr)
    
    zeros_x, zeros_r = np.zeros(nnx), np.zeros(nnr)

    poisson_rx = Poisson(xmin, xmax, nnx, rmin, rmax, nnr, 'axi_dirichlet')

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 2e-3, 0

    # interior rhs
    physical_rhs = gaussian(poisson_rx.X.reshape(-1), poisson_rx.Y.reshape(-1), ni0, x0, y0,
                        sigma_x, sigma_y)
    poisson_rx.solve(physical_rhs, zeros_x, zeros_r, zeros_r)
    poisson_rx.plot_2D(fig_dir + '2D', axi=True)
    poisson_rx.plot_1D2D(fig_dir + 'full', axi=True)