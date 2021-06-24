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
    basecase_dir = f'{os.getenv("POISSON_DIR")}/cases/superposition/'
    plot = True

    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['bcs'] = 'dirichlet'

    poisson = PoissonLinSystem(cfg)

    zero_rhs = np.zeros_like(poisson.X)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
    linear_x = np.linspace(0, 1.0, poisson.nnx)
    linear_y = np.linspace(0, 1.0, poisson.nny)

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2

    # rhs alone
    case_dir = f'{basecase_dir}rhs/'
    physical_rhs = pf.gaussian(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    pot_bcs = {'left': zeros_y, 'right': zeros_y, 'bottom': zeros_x, 'top': zeros_x}
    poisson.run_case(case_dir, physical_rhs, pot_bcs, plot)

    # dirichlet alone
    case_dir = f'{basecase_dir}laplace/'
    Vmax = 5e-3
    pot_bcs = {'left':zeros_y, 'right':Vmax * ones_y, 'bottom':Vmax * linear_x, 'top':Vmax * linear_x}
    poisson.run_case(case_dir, zero_rhs, pot_bcs, plot)

    # superposition
    poisson.run_case(basecase_dir, physical_rhs, pot_bcs, plot)
