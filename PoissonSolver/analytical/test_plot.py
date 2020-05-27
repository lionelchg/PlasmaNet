########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import sys
import os
import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve
from scipy import interpolate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_1D, plot_ax_set_1D, plot_potential
from poissonsolver.plot_comp import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff, compute_voln

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def triangle(x, x0, sigma):
    return np.maximum(0, 1 - np.abs((x - x0) / sigma))

def triangle_2D(X, Y, ampl, x0, y0, sigma_x, sigma_y):
    return ampl * triangle(X, x0, sigma_x) * triangle(Y, y0, sigma_y)

def step(x, x0, sigma):
    return (np.sign(1 - np.abs(2 * (x - x0) / sigma)) + 1) / 2

def step_2D(X, Y, ampl, x0, y0, sigma_x, sigma_y):
    return ampl * step(X, x0, sigma_x) * step(Y, y0, sigma_y)

def integral_term(x, y, Lx, Ly, voln, rhs, n, m):
    return 4 / Lx / Ly * np.sum(np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) * rhs * voln)

def fourier_coef(x, y, Lx, Ly, voln, rhs, n, m):
    return integral_term(x, y, Lx, Ly, voln, rhs, n, m) / ((n / Lx)**2 + (m / Ly)**2) / np.pi**2

def series_term(x, y, Lx, Ly, voln, rhs, n, m):
    return fourier_coef(x, y, Lx, Ly, voln, rhs, n, m) * np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly)

def sum_series(x, y, Lx, Ly, voln, rhs, N, M):
    series = np.zeros_like(x)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            series += series_term(x, y, Lx, Ly, voln, rhs, n, m)
    return series


if __name__ == '__main__':
    fig_dir = f'figures/rhs/test_plot/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    n_points = 101
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)
    voln = compute_voln(X, dx, dy)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 2e-3, 2e-3
    x0, y0 = 0.5e-2, 0.5e-2
    rhs = np.zeros(n_points ** 2)

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    # physical_rhs = triangle_2D(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    # physical_rhs = step_2D(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0

    # Scale the rhs
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    E_field = - grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)


    # Plots
    figname = fig_dir + 'solver_solution'
    plot_potential(X, Y, dx, dy, potential, n_points, n_points, figname)

    # Analytical solution but with a quadrature formula for the Fourier coefficient
    list_N = [1, 3, 5, 7]
    for N in list_N:
        M = N
        potential_th = sum_series(X, Y, Lx, Ly, voln, physical_rhs, N, M)
        E_th = - grad(potential_th, dx, dy, n_points, n_points)
        casename = 'fourier_%d_%d' % (N, M)
        figname = fig_dir + casename
        plot_potential(X, Y, dx, dy, potential_th, n_points, n_points, figname)
        plot_set_2D(X, Y, physical_rhs, potential_th, potential, E_th, E_field, f'Fourier {N:d}', figname + '_comp_solver')

