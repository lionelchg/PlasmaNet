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

def run_case(case_dir, physical_rhs, zeros_x, zeros_y, plot):
    create_dir(case_dir)
    poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
    poisson.save(case_dir)
    if plot:
        fig_dir = case_dir + 'figures/'
        create_dir(fig_dir)
        poisson.plot_2D(fig_dir + '2D')
        poisson.plot_1D2D(fig_dir + 'full')
        poisson.plot_pmodes(fig_dir + 'modes')

if __name__ == '__main__':
    basecase_dir = '../tests/cases/'
    plot = False
    xmin, xmax, nnx = 0, 0.01, 101
    ymin, ymax, nny = 0, 0.01, 101
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

    poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', 15)

    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    case_dir = f'{basecase_dir}rhs/gaussian/'
    physical_rhs = pf.gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    run_case(case_dir, physical_rhs, zeros_x, zeros_y, plot)

    case_dir = f'{basecase_dir}rhs/step/'
    physical_rhs = pf.step(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    run_case(case_dir, physical_rhs, zeros_x, zeros_y, plot)

    case_dir = f'{basecase_dir}rhs/two_gaussians/'
    x01, y01 = 0.4e-2, 0.6e-2    
    physical_rhs = pf.two_gaussians(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    ni0, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x, sigma_y) * co.e / co.epsilon_0
    run_case(case_dir, physical_rhs, zeros_x, zeros_y, plot)

    case_dir = f'{basecase_dir}rhs/random_2D/'
    physical_rhs = pf.random2D(poisson.X, poisson.Y, ni0, 16).reshape(-1) * co.e / co.epsilon_0
    run_case(case_dir, physical_rhs, zeros_x, zeros_y, plot)

    case_dir = f'{basecase_dir}rhs/sin_2D/'
    physical_rhs = pf.sin2D(poisson.X, poisson.Y, ni0, poisson.Lx, poisson.Ly, 4, 4).reshape(-1) * co.e / co.epsilon_0
    run_case(case_dir, physical_rhs, zeros_x, zeros_y, plot)
