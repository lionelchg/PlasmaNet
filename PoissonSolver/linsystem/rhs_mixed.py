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
    plot = True

    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    # 2 neumann - 2 dirichlet
    cfg['bcs'] = {'left':'neumann', 'right':'neumann', 
                    'top':'dirichlet', 'bottom':'dirichlet'}

    basecase_dir = f'{os.getenv("POISSON_DIR")}/cases/mixed/2d_2n/'
    poisson = PoissonLinSystem(cfg)
    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    pot_bcs = {'bottom':zeros_x, 'top':zeros_x}

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.6e-2, 0.5e-2
    case_dir = f'{basecase_dir}/gaussian/'
    physical_rhs = pf.gaussian(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

    case_dir = f'{basecase_dir}/dipole/'
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

    # 3 Neumann - 1 Dirichlet
    cfg['bcs'] = {'left':'neumann', 'right':'neumann', 
                    'top':'dirichlet', 'bottom':'neumann'}

    basecase_dir = f'{os.getenv("POISSON_DIR")}/cases/mixed/1d_3n/'
    poisson = PoissonLinSystem(cfg)
    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    pot_bcs = {'top':zeros_x}

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.6e-2, 0.5e-2
    case_dir = f'{basecase_dir}/gaussian/'
    physical_rhs = pf.gaussian(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

    case_dir = f'{basecase_dir}/dipole/'
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

    # 1 Neumann - 3 Dirichlet
    cfg['bcs'] = {'left':'dirichlet', 'right':'neumann', 
                    'top':'dirichlet', 'bottom':'dirichlet'}

    basecase_dir = f'{os.getenv("POISSON_DIR")}/cases/mixed/3d_1n/'
    poisson = PoissonLinSystem(cfg)
    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    pot_bcs = {'left':zeros_y, 'bottom':zeros_x, 'top':zeros_x}

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.6e-2, 0.5e-2
    case_dir = f'{basecase_dir}/gaussian/'
    physical_rhs = pf.gaussian(poisson.X, poisson.Y, 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

    case_dir = f'{basecase_dir}/dipole/'
    x01, y01 = 0.4e-2, 0.5e-2
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, 
                    [ni0, x0, y0, sigma_x, sigma_y, -ni0, x01, y01, sigma_x, sigma_y]) * co.e / co.epsilon_0
    run_case(poisson, case_dir, physical_rhs, pot_bcs, plot)

