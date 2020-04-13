########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import integrate
from poissonsolver.plot import plot_set_1D, plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.operators import grad, lapl

# Creation of directories
fig_dir = 'figures/dirichlet/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


def coef(V_u, n, x, Lx):
    return integrate.simps(V_u * np.sin(n * np.pi * x / Lx), x)

def series_term(V_u, x, y, Lx, Ly, n):
    return coef(V_u, n, x[0, :], Lx) * np.sin(n * np.pi * x / Lx) * np.sinh(n * np.pi * y / Lx) / np.sinh(n * np.pi * Ly / Lx)

def sum_series(V_u, x, y, Lx, Ly, N):
    series = np.zeros_like(x)
    for n in range(1, N + 1):
        series += series_term(V_u, x, y, Lx, Ly, n)
    return 2 / Lx * series

def series_term_exact(x, y, Lx, Ly, m):
    return np.sin((2 * m + 1) * np.pi * x / Lx) * np.sinh((2 * m + 1) * np.pi * y / Lx) / (2 * m + 1) / np.sinh((2 * m + 1) * np.pi * Ly / Lx)

def sum_series_exact(x, y, Lx, Ly, V, M):
    series = np.zeros_like(x)
    for m in range(M + 1):
        series += series_term_exact(x, y, Lx, Ly, m)
    return 4 * V / np.pi * series

if __name__ == '__main__':

    plot = True

    n_points = 64
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin
    dx = (xmax - xmin) / (n_points - 1)
    dy = (ymax - ymin) / (n_points - 1)
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    # test for dirichlet boundary conditions only (no rhs)
    potential = np.zeros((n_points, n_points))

    rhs = np.zeros(n_points ** 2)
    physical_rhs = np.zeros(n_points ** 2)

    V = 100
    linear_xy = np.linspace(xmin, xmax, n_points)
    xm = 0.5 * (xmin + xmax)
    L = xmax - xmin
    ones_bc = np.ones(n_points)
    zeros_bc = np.zeros(n_points)
    linear_bc = np.linspace(0, V, n_points)

    # Constant up
    up = V * ones_bc
    down = zeros_bc
    left = zeros_bc
    right = zeros_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    E_field = grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)
    casename = 'constant_up'
    figname = fig_dir + casename
    plot_set_1D(x, physical_rhs, potential, E_field_norm, lapl_pot, n_points, 'Solver solution 1D', figname + '_1D', no_rhs=True)
    plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Potential up', figname + '_2D', no_rhs=True)

    # Analytical solution in terms of a series
    list_M = [0, 5, 10, 20, 30]
    for M in list_M:
        potential_th = sum_series_exact(X, Y, Lx, Ly, V, M)
        E_field_th = grad(potential_th, dx, dy, n_points, n_points)
        E_field_norm_th = np.sqrt(E_field_th[0]**2 + E_field_th[1]**2)
        lapl_pot_th = lapl(potential_th, dx, dy, n_points, n_points)
        casename = 'constant_up_series_%d' % M
        figname = fig_dir + casename
        plot_set_1D(x, physical_rhs, potential_th, E_field_norm_th, lapl_pot_th, n_points, 'Potential up series M = %d' % M, figname + '_1D', no_rhs=True)
        plot_set_2D(X, Y, physical_rhs, potential_th, E_field_th, 'Potential up series M = %d' % M, figname + '_2D', no_rhs=True)

    # Analytical solution but with a quadrature formula for the Fourier coefficient
    list_N = [2 * M + 1 for M in list_M]
    for N in list_N:
        potential_th = sum_series(V * ones_bc, X, Y, Lx, Ly, N)
        E_field_th = grad(potential_th, dx, dy, n_points, n_points)
        E_field_norm_th = np.sqrt(E_field_th[0]**2 + E_field_th[1]**2)
        lapl_pot_th = lapl(potential_th, dx, dy, n_points, n_points)
        casename = 'constant_up_series_quadrature_%d' % N
        figname = fig_dir + casename
        plot_set_1D(x, physical_rhs, potential_th, E_field_norm_th, lapl_pot_th, n_points, 'Potential up series N = %d' % N, figname + '_1D', no_rhs=True)
        plot_set_2D(X, Y, physical_rhs, potential_th, E_field_th, 'Potential up series N = %d' % N, figname + '_2D', no_rhs=True)