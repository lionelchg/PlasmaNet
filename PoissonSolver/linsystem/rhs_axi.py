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

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem, run_case
from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.utils import create_dir


if __name__ == '__main__':
    basecase_dir = 'cases/axi/'
    plot = True
    with open('poisson_ls_xr.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)
    zeros_x, zeros_r = np.zeros(poisson.nnx), np.zeros(poisson.nny)

    # Boundary conditions
    pot_bcs = {'left':zeros_r, 'right':zeros_r, 'top':zeros_x}

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 2e-3, 0

    # interior rhs
    case_dir = f'{basecase_dir}rhs/gaussian/'
    physical_rhs = gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), ni0, x0, y0,
                        sigma_x, sigma_y)
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)