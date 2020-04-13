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
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_1D, plot_set_2D, plot_ax_set_1D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff, compute_voln

fig_dir = 'figures/rhs/gaussian/'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def fourier_coef(x, y, Lx, Ly, voln, rhs, n, m):
    return 4 / Lx / Ly * np.sum(np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) * rhs * voln)

def series_term(x, y, Lx, Ly, voln, rhs, n, m):
    return fourier_coef(x, y, Lx, Ly, voln, rhs, n, m) * np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) / ((n / Lx)**2 + (m / Ly)**2)

def sum_series(x, y, Lx, Ly, voln, rhs, N, M):
    series = np.zeros_like(x)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            series += series_term(x, y, Lx, Ly, voln, rhs, n, m)
    return series / np.pi**2


if __name__ == '__main__':
    n_points = 101
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    Lx, Ly = xmax - xmin, ymax - ymin
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-1, 1e-1
    x0, y0 = 0.5, 0.5
    rhs = np.zeros(n_points ** 2)

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    E_field = grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)


    # Plots
    figname = fig_dir + 'solver_solution'
    plot_set_1D(x, physical_rhs, potential, E_field_norm, lapl_pot, n_points, 'Solver solution 1D', figname + '_1D')
    plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Solver solution 2D', figname + '_2D')


    # Analytical solution but with a quadrature formula for the Fourier coefficient
    list_N = [1, 3, 5, 9, 15]
    for N in list_N:
        M = N
        voln = compute_voln(X, dx, dy)
        potential_th = sum_series(X, Y, Lx, Ly, voln, physical_rhs, N, M)
        E_field_th = grad(potential_th, dx, dy, n_points, n_points)
        E_field_norm_th = np.sqrt(E_field_th[0]**2 + E_field_th[1]**2)
        lapl_pot_th = lapl(potential_th, dx, dy, n_points, n_points)
        casename = 'constant_up_series_quadrature_%d_%d' % (N, M)
        figname = fig_dir + casename
        plot_set_1D(x, physical_rhs, potential_th, E_field_norm_th, lapl_pot_th, n_points, 'Potential up series N = %d M = %d' % (N, M), figname + '_1D')
        plot_set_2D(X, Y, physical_rhs, potential_th, E_field_th, 'Potential up series N = %d M = %d' % (N, M), figname + '_2D')