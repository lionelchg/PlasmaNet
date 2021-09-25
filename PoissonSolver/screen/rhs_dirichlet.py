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
from pathlib import Path

import PlasmaNet.common.profiles as pf
from PlasmaNet.poissonscreensolver.screenpoisson_ls import ScreenPoissonLinSystem

if __name__ == '__main__':
    plot = True

    with open('cart.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)
    cfg['bcs'] = 'dirichlet'
    coeffs = [0.0, 1e1, 1e2, 1e3, 1e4]

    for icoeff, coeff in enumerate(coeffs):
        cfg['coeff'] = coeff
        basecase_dir = Path(f'cases/dirichlet/rhs/coeff_{icoeff:d}')

        poisson = ScreenPoissonLinSystem(cfg)

        zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)

        pot_bcs = {'left': zeros_y, 'right': zeros_y, 'bottom': zeros_x, 'top': zeros_x}

        ni0 = 1e11
        sigma_x, sigma_y = 1e-3, 1e-3
        x0, y0 = 0.5e-2, 0.5e-2
        case_dir = basecase_dir / 'gaussian/'
        physical_rhs = pf.gaussian(poisson.X, poisson.Y,
                        ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
        poisson.run_case(case_dir, physical_rhs, pot_bcs, plot)

        case_dir = basecase_dir / 'two_gaussians/'
        x0, y0 = 0.6e-2, 0.5e-2
        x01, y01 = 0.4e-2, 0.5e-2
        physical_rhs = pf.two_gaussians(poisson.X, poisson.Y,
                        ni0, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
        poisson.run_case(case_dir, physical_rhs, pot_bcs, plot)

        case_dir = basecase_dir / 'random_2D/'
        physical_rhs = pf.random2D(poisson.X, poisson.Y, ni0, 16) * co.e / co.epsilon_0
        poisson.run_case(case_dir, physical_rhs, pot_bcs, plot)

