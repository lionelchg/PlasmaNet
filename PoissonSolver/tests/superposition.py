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

from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import Poisson
import PlasmaNet.common.profiles as pf

if __name__ == '__main__':
    xmin, xmax, nnx = 0, 0.01, 101
    ymin, ymax, nny = 0, 0.01, 101
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)
    ones_x, ones_y = np.ones(nnx), np.ones(nny)
    linear_x, linear_y = np.linspace(0.0, 1.0, nnx), np.linspace(0.0, 1.0, nny)

    poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', 15)

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2

    fig_dir = 'figures/superposition/rhs/'
    create_dir(fig_dir)
    physical_rhs = pf.gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
    poisson.plot_2D(fig_dir + '2D')
    poisson.plot_1D2D(fig_dir + 'full')
    poisson.plot_pmodes(fig_dir + 'modes')

    fig_dir = 'figures/superposition/linear_pot_x/'
    create_dir(fig_dir)
    Vmax = 5e-3
    poisson.solve(np.zeros_like(poisson.X).reshape(-1), 
                        Vmax * linear_x, Vmax * linear_x, zeros_y, Vmax * ones_y)
    poisson.plot_2D(fig_dir + '2D')
    poisson.plot_1D2D(fig_dir + 'full')

    fig_dir = 'figures/superposition/'
    poisson.solve(physical_rhs, 
                        Vmax * linear_x, Vmax * linear_x, zeros_y, Vmax * ones_y)
    poisson.plot_2D(fig_dir + '2D')
    poisson.plot_1D2D(fig_dir + 'full')
