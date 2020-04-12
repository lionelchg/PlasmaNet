########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as co
from scipy import interpolate
from scipy.sparse.linalg import spsolve

from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc

colormap = 'RdBu'

fig_dir = 'figures/random_rhs/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


if __name__ == '__main__':

    n_points = 64
    n_res = 8
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)

    n_lower = int(n_points / n_res)
    x_lower, y_lower = np.linspace(xmin, xmax, n_lower), np.linspace(ymin, ymax, n_lower)
    X_lower, Y_lower = np.meshgrid(x_lower, y_lower)
    z_lower = 2 * np.random.random((n_lower, n_lower)) - 1
    f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')

    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)
    z = f(x, y)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(14, 7))
    CS1 = ax1.contourf(X_lower, Y_lower, z_lower, 100, cmap=colormap)
    cbar1 = fig.colorbar(CS1, pad = 0.05, fraction=0.08, ax=ax1, aspect=5)
    cbar1.ax.set_ylabel('Under-resolved')
    ax1.set_aspect("equal")
    CS2 = ax2.contourf(X, Y, z, 100, cmap=colormap)
    cbar2 = fig.colorbar(CS2, pad = 0.05, fraction=0.08, ax=ax2, aspect=5)
    cbar2.ax.set_ylabel('Bicubic random')
    ax2.set_aspect("equal")
    plt.savefig(fig_dir + 'bicubic', bbox_inches='tight')

    # creating the rhs
    ni0 = 1e18
    rhs = np.zeros(n_points**2)

    #interior rhs
    physical_rhs = ni0 * z.reshape(-1) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx**2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    E_field = grad(potential, dx, dy, n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    figname = fig_dir + 'random_2D'
    plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Random RHS', figname)