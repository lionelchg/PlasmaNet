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
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf

if __name__ == '__main__':
    basecase_dir = f'{os.getenv("POISSON_DIR")}/cases/perio/'
    plot = True

    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['bcs'] = 'perio'

    poisson = PoissonLinSystem(cfg)
    pot_bcs = {}

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    case_dir = f'{basecase_dir}dipole/'
    x0, y0 = 0.6e-2, 0.5e-2
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, pot_bcs, plot)

    case_dir = f'{basecase_dir}sin_2D_4/'
    physical_rhs = pf.sin2D(poisson.X, poisson.Y, ni0, poisson.Lx, poisson.Ly, 4, 4) * co.e / co.epsilon_0
    poisson.run_case(case_dir, physical_rhs, pot_bcs, plot)
