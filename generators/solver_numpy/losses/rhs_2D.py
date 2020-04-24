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
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch

from poissonsolver.operators import lapl, grad, derivative
from poissonsolver.plot import plot_set_1D, plot_set_2D, plot_ax_set_1D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff, compute_voln, func_energy, func_energy_torch
import poissonsolver.operators_torch as optorch

from losses import plot_potential, plot_ax_3D

figs_dir = 'figures/gaussian/'

if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

print_bool = False

mpl.rcParams['lines.linewidth'] = 2

loss_list = ['Energy', 'Points', 'Lapl', 'Electric']
grid_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]

def triangle(x, L, p):
    return (1 - np.abs(2 * (x - L/2) / L))**p

def bell(x, L, p):
    return (1 - np.abs((x - L/2) * 2 / L)**p)

def fourier_mode(X, Y, Lx, Ly, n, m):
    return np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly)

def trial_triangle(X, Y, Lx, Ly, ampl, p):
    return ampl * triangle(X, Lx, p) * triangle(Y, Ly, p)

def trial_bell(X, Y, Lx, Ly, ampl, p):
    return ampl * bell(X, Lx, p) * bell(Y, Ly, p)

def trial_fourier(X, Y, Lx, Ly, ampl1, ampl2):
    return ampl1 * fourier_mode(X, Y, Lx, Ly, 1, 1) + ampl2 * (fourier_mode(X, Y, Lx, Ly, 3, 1) + fourier_mode(X, Y, Lx, Ly, 1, 3))

trialfunctions = dict([('triangle', trial_triangle), ('bell', trial_bell), ('fourier', trial_fourier)])

def losses_2D(ampl_range, p_range, figname, trial):

    Ampl_range, P_range = np.meshgrid(ampl_range, p_range)
    losses = np.zeros((4, Ampl_range.shape[0], Ampl_range.shape[1]))
    function = trialfunctions[trial]

    for i, ampl_potential in enumerate(ampl_range):
        for j, p_potential in enumerate(p_range):
            potential = function(X, Y, Lx, Ly, ampl_potential, p_potential)
            E_field, E_field_norm, lapl_pot, energy, points_loss, lapl_loss, elec_loss = compute_values(potential)
            losses[:, j, i] = np.array([energy, points_loss, lapl_loss, elec_loss])

    min_coord = np.zeros((4, 3))
    for i in range(4):
        imin = np.argmin(losses[i])
        loss_min = np.min(losses[i])
        ampl_potential = Ampl_range.reshape(-1)[imin]
        p_potential = P_range.reshape(-1)[imin]

        potential = function(X, Y, Lx, Ly, ampl_potential, p_potential)

        title =  'A = %.2f p = %.2f Losses = %.3e %.3e %.3e %.3e' % (ampl_potential, p_potential, losses[0].reshape(-1)[imin],
                    losses[1].reshape(-1)[imin], losses[2].reshape(-1)[imin], losses[3].reshape(-1)[imin])
        plot_potential(X, Y, dx, dy, potential, n_points, fig_dir + 'min_trial_' + loss_list[i], title)
        min_coord[i, :] = np.array([ampl_potential, p_potential, imin])

    fig = plt.figure(figsize=(10, 10))

    for i in range(4):
        ax = plot_ax_3D(fig, i + 1, Ampl_range, P_range, losses[i], loss_list[i])
        for j in range(4):
            ax.scatter(min_coord[j, 0], min_coord[j, 1], losses[i].reshape(-1)[int(min_coord[j, 2])], 'ro')
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')

def compute_values(potential):
    """ Computes the electric field and the laplacian of the potential as well as losses """
    E_field = - grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)

    functional_energy = func_energy(potential, E_field, physical_rhs, voln) - functional_energy_target
    points_loss = np.sqrt(np.sum((potential - potential_target)**2)) / n_points**2
    interior_diff = lapl_diff(potential, physical_rhs, dx, dy, n_points, n_points)
    lapl_loss = np.sqrt(np.sum(interior_diff**2)) / n_points**2
    elec_loss = np.sqrt(np.sum((E_field_norm - E_field_norm_target)**2)) / n_points**2

    return E_field, E_field_norm, lapl_pot, functional_energy, points_loss, lapl_loss, elec_loss

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

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
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    rhs = np.zeros(n_points ** 2)

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential_target = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)

    # Computing target values related to the potential
    E_field_target = - grad(potential_target, dx, dy, n_points, n_points)
    E_field_norm_target = np.sqrt(E_field_target[0]**2 + E_field_target[1]**2)
    lapl_pot_target = lapl(potential_target, dx, dy, n_points, n_points)
    functional_energy_target = func_energy(potential_target, E_field_target, physical_rhs, voln)

    # Plots
    figname = figs_dir + 'solver_solution'
    plot_potential(X, Y, dx, dy, potential_target, n_points, figname)
    
    #######################
    #
    #    1D Loss study
    #
    #######################
    typeloss = '2D/'

    # Bell
    typetrial = 'bell/'
    fig_dir = figs_dir + typeloss + typetrial
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    ampl_range = np.linspace(1, 300, 300)
    p_range = np.linspace(0.1, 1.5, 30)

    losses_2D(ampl_range, p_range, fig_dir + 'losses', 'bell')

    # Triangle
    typetrial = 'triangle/'
    fig_dir = figs_dir + typeloss + typetrial
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    ampl_range = np.linspace(1, 300, 150)
    p_range = np.linspace(0.5, 2.5, 41)

    losses_2D(ampl_range, p_range, fig_dir + 'losses', 'triangle')

    # Fourier
    typetrial = 'fourier/'
    fig_dir = figs_dir + typeloss + typetrial
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    ampl_range_0 = np.linspace(60, 160, 101)
    ampl_range_1 = np.linspace(-30, -10, 21)

    losses_2D(ampl_range_0, ampl_range_1, fig_dir + 'losses', 'fourier')
