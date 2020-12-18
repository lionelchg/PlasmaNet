########################################################################################################################
#                                                                                                                      #
#                                        2D Poisson analytical solution tests                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import scipy.constants as co
from scipy import interpolate

from poissonsolver.poisson import Poisson
from poissonsolver.analytical import PoissonAnalytical
from poissonsolver.funcs import gaussian
from poissonsolver.utils import create_dir

@profile
def main():
    fig_dir = 'figures/lsys_vs_th/'
    create_dir(fig_dir)

    nsolv = 3
        
    xmin, xmax, nnx = 0, 0.01, 601
    ymin, ymax, nny = 0, 0.01, 601
    nmax_rhs, mmax_rhs = 16, 16
    nmax_d = 10
    dx, dy = (xmax - xmin) / (nnx - 1), (ymax - ymin) / (nny - 1)
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)

    zeros_x, zeros_y = np.zeros(nnx), np.zeros(nny)
    ones_x, ones_y = np.ones(nnx), np.ones(nny)

    # Declaration of poisson solver (linear system)
    poisson = Poisson(xmin, xmax, nnx, ymin, ymax, nny, 'cart_dirichlet')

    # creating the rhs
    ni0 = 1e11
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2

    # interior rhs
    physical_rhs = gaussian(poisson.X, poisson.Y, ni0, x0, y0, 
                            sigma_x, sigma_y) * co.e / co.epsilon_0

    for i in range(nsolv):
        poisson.solve(physical_rhs.reshape(-1), zeros_x, zeros_x, zeros_y, zeros_y)

    poisson.plot_2D(fig_dir + '2D')
    poisson.plot_1D2D(fig_dir + 'full')

    # Declaration of class Poisson Analytical
    poisson_th = PoissonAnalytical(xmin, xmax, nnx, ymin, ymax, nny, nmax_rhs, mmax_rhs, nmax_d)

    # Solve rhs problem
    for i in range(nsolv):
        poisson_th.compute_sol(physical_rhs, zeros_y, zeros_y, zeros_x, zeros_x)

    poisson_th.plot_2D(fig_dir + '2D_th')
    poisson_th.plot_1D2D(fig_dir + 'full_th')

if __name__ == '__main__':
    main()