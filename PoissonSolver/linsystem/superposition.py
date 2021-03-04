########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import yaml
import numpy as np
import scipy.constants as co

from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf

if __name__ == '__main__':
    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
    linear_x = np.linspace(0, 1.0, poisson.nnx)
    linear_y = np.linspace(0, 1.0, poisson.nny)

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
