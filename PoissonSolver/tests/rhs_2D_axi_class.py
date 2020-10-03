########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
import numpy as np
import scipy.constants as co
import copy

from poissonsolver.poisson import Poisson

fig_dir = 'figures/rhs_2D_axi_class/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


if __name__ == '__main__':
    xmin, xmax, nnx = 0, 4e-3, 401
    rmin, rmax, nnr = 0, 1e-3, 101
    x, r = np.linspace(xmin, xmax, nnx), np.linspace(rmin, rmax, nnr)
    
    zeros_x, zeros_r = np.zeros(nnx), np.zeros(nnr)

    poisson = Poisson(xmin, xmax, nnx, rmin, rmax, nnr, 'axi_dirichlet')

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 2e-3, 0
    rhs = np.zeros(nnx * nnr)

    # interior rhs
    physical_rhs = gaussian(poisson.X.reshape(-1), poisson.Y.reshape(-1), ni0, x0, y0,
                        sigma_x, sigma_y)
    
    poisson.solve(physical_rhs, zeros_x, zeros_r, zeros_r)

    poisson.plot_potential(fig_dir + 'cylindrical_pot')