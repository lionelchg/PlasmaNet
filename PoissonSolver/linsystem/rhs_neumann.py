########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co

from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem, run_case
import PlasmaNet.common.profiles as pf

if __name__ == '__main__':
    basecase_dir = 'cases/neumann/'
    plot = True

    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['bcs'] = 'neumann'

    poisson = PoissonLinSystem(cfg)
    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    pot_bcs = {}

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.6e-2, 0.5e-2

    case_dir = f'{basecase_dir}/dipole/'
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

    case_dir = f'{basecase_dir}/quadrupole/'
    physical_rhs = pf.gaussians(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    [ni0, 0.4e-2, 0.4e-2, sigma_x, sigma_y, 
                    -ni0, 0.6e-2, 0.4e-2, sigma_x, sigma_y,
                    ni0, 0.6e-2, 0.6e-2, sigma_x, sigma_y,
                    -ni0, 0.4e-2, 0.6e-2, sigma_x, sigma_y]) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)