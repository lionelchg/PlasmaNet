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

from scipy.sparse.linalg import spsolve
from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_2D, plot_potential, plot_lapl_rhs
from poissonsolver.linsystem import matrix_cart, matrix_axisym, dirichlet_bc_axi
from poissonsolver.postproc import lapl_diff

fig_dir = 'figures/rhs_2D_axi/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


if __name__ == '__main__':
    xmin, xmax, nx = 0, 4e-3, 401
    rmin, rmax, nr = 0, 1e-3, 101
    dx, dr = (xmax - xmin) / (nx - 1), (rmax - rmin) / (nr - 1)
    x, r = np.linspace(xmin, xmax, nx), np.linspace(rmin, rmax, nr)

    X, R = np.meshgrid(x, r)

    scale = dx * dr

    physical_rhs = np.zeros_like(X)

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 2e-4, 0
    rhs = np.zeros(nx * nr)

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), R.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0

    # Cartesian resolution
    A = matrix_cart(dx, dr, nx, nr, scale)
    rhs = - physical_rhs
    zeros_x, zeros_r = np.zeros(nx), np.zeros(nr)
    dirichlet_bc_axi(rhs, nx, nr, zeros_x, zeros_r, zeros_r)
    potential = spsolve(A, rhs).reshape(nr, nx)

    # Axisymmetric resolution
    R_nodes = copy.deepcopy(R)
    R_nodes[0] = dr / 4
    A = matrix_axisym(dx, dr, nx, nr, R_nodes, scale)
    rhs = - physical_rhs
    dirichlet_bc_axi(rhs, nx, nr, zeros_x, zeros_r, zeros_r)
    potential_axi = spsolve(A, rhs).reshape(nr, nx)

    # Plots
    physical_rhs = physical_rhs.reshape(nr, nx)
    plot_potential(X, R, dx, dr, potential, nx, nr, fig_dir + 'cartesian_pot')
    plot_potential(X, R, dx, dr, potential_axi, nx, nr, fig_dir + 'cylindrical_pot', r=R_nodes)
    plot_lapl_rhs(X, R, dx, dr, potential_axi, physical_rhs, nx, nr, fig_dir + 'comp_lapl_rhs_cyl', r=R_nodes)