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

from scipy.sparse.linalg import spsolve, isolve
from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_2D, plot_potential, plot_lapl_rhs
from poissonsolver.linsystem import matrix_cart, matrix_axisym, dirichlet_bc_axi, laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff

fig_dir = 'figures/rhs_2D_axi/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def compute_voln(X, dx, dy):
    """ Computes the nodal volume associated to each node (i, j) """
    voln = np.ones_like(X) * dx * dy
    voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
        dx * dy / 2, dx * dy / 2, dx * dy / 2, dx * dy / 2
    voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
        dx * dy / 4, dx * dy / 4, dx * dy / 4, dx * dy / 4
    return voln

if __name__ == '__main__':
    xmin, xmax, nx = 0, 0.001, 101
    rmin, rmax, nr = 0, 0.001, 101
    dx, dr = (xmax - xmin) / (nx - 1), (rmax - rmin) / (nr - 1)
    x, r = np.linspace(xmin, xmax, nx), np.linspace(rmin, rmax, nr)

    X, R = np.meshgrid(x, r)

    scale = dx * dr

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-1, 1e-1
    x0, y0 = 0.5, 0
    rhs = np.zeros(nx * nr)

    # interior rhs
    # physical_rhs = gaussian(X.reshape(-1), R.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    physical_rhs = np.zeros_like(X.reshape(-1))

    # Background electric field
    backE = 100e2
    up = - x * backE
    down = - x * backE
    left = np.zeros_like(r)
    right = - np.ones_like(r) * backE * xmax

    # Basic dirichlet square
    A = laplace_square_matrix(nx)
    rhs = - physical_rhs * dx**2
    dirichlet_bc(rhs, nx, up, down, left, right)
    potential_fulld = spsolve(A, rhs).reshape(nr, nx)
    # potential_fulld = isolve.cgs(A, rhs, tol=1e-5)[0].reshape(nr, nx)

    # Cartesian resolution
    A = matrix_cart(dx, dr, nx, nr, scale)
    rhs = - physical_rhs * scale
    zeros_x, zeros_r = np.zeros(nx), np.zeros(nr)
    dirichlet_bc_axi(rhs, nx, nr, up, left, right)
    potential = spsolve(A, rhs).reshape(nr, nx)

    # Axisymmetric resolution
    R_nodes = copy.deepcopy(R)
    R_nodes[0] = dr / 4
    A = matrix_axisym(dx, dr, nx, nr, R_nodes, scale)
    rhs = - physical_rhs * scale
    dirichlet_bc_axi(rhs, nx, nr, up, left, right)
    potential_axi = spsolve(A, rhs).reshape(nr, nx)

    # Plots
    physical_rhs = physical_rhs.reshape(nr, nx)
    plot_potential(X, R, dx, dr, potential_fulld, nx, nr, fig_dir + 'cartesian_pot_backE_fulld')
    plot_potential(X, R, dx, dr, potential, nx, nr, fig_dir + 'cartesian_pot_backE')
    plot_potential(X, R, dx, dr, potential_axi, nx, nr, fig_dir + 'cylindrical_pot_backE', r=R_nodes)
    # plot_lapl_rhs(X, R, dx, dr, potential_axi, physical_rhs, nx, nr, fig_dir + 'comp_lapl_rhs_cyl_backE', r=R_nodes)

    # Comparison and prints
    print('Imposed potential')
    print(down)
    print(left)
    print(up)
    print(right)
    print('Full dirichlet potential:')
    print(potential_fulld[0, :])
    print(potential_fulld[:, 0])
    print(potential_fulld[-1, :])
    print(potential_fulld[:, -1])
    print('Cartesian potential:')
    print(potential[0, :])
    print(potential[:, 0])
    print(potential[-1, :])
    print(potential[:, -1])
    print('Axisymmetric potential:')
    print(potential_axi[0, :])
    print(potential_axi[:, 0])
    print(potential_axi[-1, :])
    print(potential_axi[:, -1])
