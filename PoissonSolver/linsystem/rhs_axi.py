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

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.utils import create_dir


if __name__ == '__main__':
    fig_dir = 'figures/rhs/axi/'
    create_dir(fig_dir)

    with open('poisson_ls_xr.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)
    zeros_x, zeros_r = np.zeros(poisson.nnx), np.zeros(poisson.nny)

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 2e-3, 0

    # interior rhs
    physical_rhs = gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), ni0, x0, y0,
                        sigma_x, sigma_y)
    poisson.solve(physical_rhs, zeros_x, zeros_r, zeros_r)
    poisson.plot_2D(fig_dir + '2D', geom='xr')
    poisson.plot_1D2D(fig_dir + 'full', geom='xr')
