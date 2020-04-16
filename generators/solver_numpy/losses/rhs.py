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

fig_dir = 'figures/rhs/gaussian/'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

print_bool = False

mpl.rcParams['lines.linewidth'] = 2

def compute_values(ampl_potential, trial_function, print_bool=False):
    potential = ampl_potential * trial_function
    E_field = - grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    functional_energy = func_energy(potential, E_field, physical_rhs, voln) - functional_energy_target
    points_loss = np.sum((potential - potential_target)**2) / n_points**2
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)
    interior_diff = lapl_diff(potential, physical_rhs, dx, dy, n_points, n_points)
    lapl_loss = np.sum(interior_diff**2) / n_points**2
    elec_loss = np.sum((E_field_norm - E_field_norm_target)**2) / n_points**2

    info_test = 'A = %.2e, energy = %.4e - points_loss = %.4e - lapl_loss = %.4e - elec_loss = %.4e' \
            % (ampl_potential, functional_energy, points_loss, lapl_loss, elec_loss)
    if print_bool: print(info_test)

    return potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, lapl_loss, elec_loss, info_test

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def fourier_coef(x, y, Lx, Ly, voln, rhs, n, m):
    return 4 / Lx / Ly * np.sum(np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) * rhs * voln)

def series_term(x, y, Lx, Ly, voln, rhs, n, m):
    return fourier_coef(x, y, Lx, Ly, voln, rhs, n, m) * np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) / ((n / Lx)**2 + (m / Ly)**2) / np.pi**2

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
    E_field_target = - grad(potential_target, dx, dy, n_points, n_points)
    E_field_norm_target = np.sqrt(E_field_target[0]**2 + E_field_target[1]**2)
    lapl_pot_target = lapl(potential_target, dx, dy, n_points, n_points)
    functional_energy_target = func_energy(potential_target, E_field_target, physical_rhs, voln)

    # Plots
    figname = fig_dir + 'solver_solution'
    plot_set_1D(x, physical_rhs, potential_target, E_field_norm_target, lapl_pot_target, n_points, 'Solver solution 1D', figname + '_1D')
    plot_set_2D(X, Y, physical_rhs, potential_target, E_field_target, 'Solver solution 2D', figname + '_2D')


    # Analytical solution but with a quadrature formula for the Fourier coefficient
    N = 1
    M = N
    potential_th = sum_series(X, Y, Lx, Ly, voln, physical_rhs, N, M)
    E_field_th = - grad(potential_th, dx, dy, n_points, n_points)
    E_field_norm_th = np.sqrt(E_field_th[0]**2 + E_field_th[1]**2)
    functional_energy_th = func_energy(potential_th, E_field_th, physical_rhs, voln)
    lapl_pot_th = lapl(potential_th, dx, dy, n_points, n_points)
    casename = 'gaussian_series_quadrature_%d_%d' % (N, M)
    figname = fig_dir + casename
    plot_set_1D(x, physical_rhs, potential_th, E_field_norm_th, lapl_pot_th, n_points, 'Potential up series N = %d M = %d' % (N, M), figname + '_1D')
    plot_set_2D(X, Y, physical_rhs, potential_th, E_field_th, 'Potential up series N = %d M = %d' % (N, M), figname + '_2D')

    # Trial function (the first harmonic)
    trial_function = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
    coef_target = fourier_coef(X, Y, Lx, Ly, voln, physical_rhs, 1, 1) / ((1 / Lx)**2 + (1 / Ly)**2) / np.pi**2
    print('coef target = %.4e' % coef_target)

    coef_range = np.linspace(1, 200, 200)
    list_loss = []
    for ampl_potential in coef_range:
        potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, \
            lapl_loss, elec_loss, info_test = compute_values(ampl_potential, trial_function)
        list_loss.append([functional_energy, points_loss, lapl_loss, elec_loss])
    losses_1D = np.array(list_loss)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes[0][0].plot(coef_range, losses_1D[:, 0])
    axes[0][0].set_title('Energy')
    axes[0][0].grid()
    axes[0][1].plot(coef_range, losses_1D[:, 1])
    axes[0][1].set_title('Points Loss')
    axes[0][1].grid()
    axes[1][0].plot(coef_range, losses_1D[:, 2])
    axes[1][0].set_title('Lapl Loss')
    axes[1][0].grid()
    axes[1][1].plot(coef_range, losses_1D[:, 3])
    axes[1][1].set_title('Electric Loss')
    axes[1][1].grid()
    plt.tight_layout()
    plt.savefig(fig_dir + 'losses_1D', bbox_inches='tight')

    energy_derivative = derivative(losses_1D[:, 0], coef_range, 1)
    f = interpolate.interp1d(coef_range, energy_derivative)

    #Conversion to torch tensors
    physical_rhs = torch.from_numpy(physical_rhs)
    X, Y, voln = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(voln)
    trial_function = torch.sin(np.pi * X / Lx) * torch.sin(np.pi * Y / Ly)

    ampl_potential = 50.0
    eta = 0.1
    for i in range(10):
        print('Step %d' % (i + 1))
        v = torch.tensor([ampl_potential], requires_grad=True)
        potential = v * trial_function
        E_field = optorch.grad(potential, dx, dy, n_points, n_points)
        functional_energy = func_energy_torch(potential, E_field, physical_rhs, voln) - functional_energy_target
        functional_energy.backward()
        print('target grad = %.2e - torch grad = %.2e - ratio = %.2e ' % (f(ampl_potential), v.grad.data, f(ampl_potential) / v.grad.data))
        print('A = %.4e, I(phi) = %.4e' % (ampl_potential, functional_energy))
        ampl_potential -= eta * v.grad.data

    ampl_potential = 200.0
    eta = 0.1
    for i in range(10):
        print('Step %d' % (i + 1))
        v = torch.tensor([ampl_potential], requires_grad=True)
        potential = v * trial_function
        E_field = optorch.grad(potential, dx, dy, n_points, n_points)
        functional_energy = func_energy_torch(potential, E_field, physical_rhs, voln) - functional_energy_target
        functional_energy.backward()
        print('target grad = %.2e - torch grad = %.2e - ratio = %.2e ' % (f(ampl_potential), v.grad.data, f(ampl_potential) / v.grad.data))
        print('A = %.4e, I(phi) = %.4e' % (ampl_potential, functional_energy))
        ampl_potential -= eta * v.grad.data

