########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import yaml

from PlasmaNet.common.profiles import random1D, random2D
from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem, run_case

if __name__ == '__main__':
    basecase_dir = '../tests/cases/dirichlet/laplace/'
    plot = True

    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['bcs'] = 'dirichlet'

    poisson = PoissonLinSystem(cfg)

    zero_rhs = np.zeros_like(poisson.X).reshape(-1)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
    linear_x = np.linspace(0, 1.0, poisson.nnx)
    linear_y = np.linspace(0, 1.0, poisson.nny)

    case_dir = f'{basecase_dir}random_left/'
    random_y = random1D(poisson.Y[:, 0], 100.0, 4)
    pot_bcs = {'left':random_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
    poisson.run_case(case_dir, zero_rhs, pot_bcs, plot)
    np.save(case_dir + 'random_left', random_y)

    case_dir = f'{basecase_dir}constant_left/'
    pot_bcs = {'left':ones_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
    poisson.run_case(case_dir, zero_rhs, pot_bcs, plot)

    case_dir = f'{basecase_dir}linear_pot_x/'
    Vmax = 100.0
    pot_bcs = {'left':zeros_y, 'right':Vmax * ones_y, 'bottom':Vmax * linear_x, 'top':Vmax * linear_x}
    poisson.run_case(case_dir, zero_rhs, pot_bcs, plot)