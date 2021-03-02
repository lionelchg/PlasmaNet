########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import spsolve

from PlasmaNet.common.profiles import random1D, random2D
from PlasmaNet.common.utils import create_dir
from PlasmaNet.poissonsolver.poisson import Poisson

def run_case(case_dir, bottom, up, left, right, plot):
    create_dir(case_dir)
    poisson.solve(np.zeros_like(poisson.X).reshape(-1), bottom, up, left, right)
    poisson.save(case_dir)
    if plot:
        fig_dir = case_dir + 'figures/'
        create_dir(fig_dir)
        poisson.plot_2D(fig_dir + '2D')
        poisson.plot_1D2D(fig_dir + 'full')

if __name__ == '__main__':
    basecase_dir = '../tests/cases/'
    plot = False
    xmin, xmax, nnx = 0, 0.01, 101
    ymin, ymax, nny = 0, 0.01, 101
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)
    ones_x, ones_y = np.ones(nnx), np.ones(nny)
    linear_x, linear_y = np.linspace(0.0, 1.0, nnx), np.linspace(0.0, 1.0, nny)

    poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', 15)

    case_dir = f'{basecase_dir}dirichlet/random_left/'
    random_y = random1D(y, 100.0, 4)
    run_case(case_dir, zeros_x, zeros_x, random_y, zeros_y, plot)
    np.save(case_dir + 'random_left', random_y)

    case_dir = f'{basecase_dir}dirichlet/constant_left/'
    run_case(case_dir, zeros_x, zeros_x, ones_y, zeros_y, plot)

    case_dir = f'{basecase_dir}dirichlet/linear_pot_x/'
    Vmax = 100.0
    run_case(case_dir, Vmax * linear_x, Vmax * linear_x, zeros_y, Vmax * ones_y, plot)