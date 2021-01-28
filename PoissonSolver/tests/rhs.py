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

from poissonsolver.poisson import Poisson
from poissonsolver.funcs import gaussian
import cfdsolver.scalar.init as init_funcs

fig_dir = 'figures/rhs_2D/triangle/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

if __name__ == '__main__':
    xmin, xmax, nnx = 0, 0.01, 101
    ymin, ymax, nny = 0, 0.01, 101
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)

    poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet', 15)

    # creating the rhs
    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2

    # # interior rhs
    # physical_rhs = gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
    #                 ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    
    physical_rhs = init_funcs.triangle(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
                    ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0

    # x01, y01 = 0.4e-2, 0.5e-2    
    # physical_rhs = init_funcs.two_gaussians(poisson.X.reshape(-1), poisson.Y.reshape(-1), 
    #                 ni0, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x, sigma_y)
    
    poisson.solve(physical_rhs, zeros_x, zeros_x, zeros_y, zeros_y)
    poisson.plot_2D(fig_dir + '2D')
    poisson.plot_1D2D(fig_dir + 'full')
    poisson.plot_pmodes(fig_dir + 'modes')