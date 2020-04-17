########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve

from operators import lapl, grad
import operators_torch as optorch
from plot import plot_fig, plot_fig_scalar, plot_vector_arrow, plot_fig_list
from poisson_2D_FD import laplace_square_matrix, dirichlet_bc, lapl_diff, compute_voln,\
                        func_energy, func_energy_torch
import matplotlib.pyplot as plt
import torch

from mpl_toolkits.mplot3d import Axes3D

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def torch_gaussian(x, y, v):
    return v[0] * torch.exp(-((x - 5e-3) / v[1]) ** 2 - ((y - 5e-3) / v[1]) ** 2)

def cosine_hill(x, y, amplitude, x0, L, power):
    r = np.sqrt((x - x0)**2 + (y - x0)**2)
    return amplitude * ((1 + np.cos(2 * np.pi / L * r)) / 2)**2

if __name__ == '__main__':
    fig_dir = 'functional/'
    n_points = 128
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    L = xmax - xmin
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
    electric_field_target = grad(potential_target, dx, dy, n_points, n_points)
    electric_field_norm_target = np.sqrt(electric_field_target[0]**2 + electric_field_target[1]**2)
    field_energy_target = co.epsilon_0 / 2 * (electric_field_target[0]**2 + electric_field_target[1]**2)
    potential_energy_target = physical_rhs * co.epsilon_0 * potential
    interior_diff_target = lapl_diff(potential_target, physical_rhs, dx, dy, n_points, n_points)
    functional_energy_target = func_energy(potential_target, electric_field_target, physical_rhs, voln)
    print('I(phi)_min = %.4e' % functional_energy_target)

    # # Plots
    # plot_fig(X, Y, potential, physical_rhs, name=fig_dir + 'gauss_', nit=1)
    # plot_fig_scalar(X, Y, interior_diff, 'Absolute difference',
    #                 fig_dir + 'gauss_abs_diff', colormap='Blues')
    # plot_fig_list(X, Y, [field_energy, potential_energy, field_energy - potential_energy],
    #  ['Field Energy', 'Potential Energy', 'Difference'], fig_dir + 'gauss_energies')
    # plot_vector_arrow(X, Y, electric_field, "Electric field", fig_dir + "gauss_electric_field")

    # Try iteration to see the best fitting gaussian
    fig, axes = plt.subplots(ncols=2, figsize=(10, 7))
    axes[0].plot(x, potential_target[int(n_points / 2), :], label='True potential')
    axes[1].plot(x, electric_field_norm_target[int(n_points / 2), :], label='True E_field')

    ampl_potential = 130
    sigma_pot = 2.51e-3
    potential = gaussian(X, Y, ampl_potential, x0, y0, sigma_pot, sigma_pot)
    potential[:, 0], potential[0, :], potential[:, -1], potential[-1, :] = 0, 0, 0 ,0
    electric_field = grad(potential, dx, dy, n_points, n_points)
    electric_field_norm = np.sqrt(electric_field[0]**2 + electric_field[1]**2)
    functional_energy = func_energy(potential, electric_field, physical_rhs, voln)
    print('A = %.2e, sigma = %.2e - energy = %.4e' % (ampl_potential, sigma_pot, functional_energy))
    axes[0].plot(x, potential[int(n_points / 2), :])
    axes[0].legend()
    axes[1].plot(x, electric_field_norm[int(n_points / 2), :])
    axes[1].legend()
    plt.suptitle('A = %.2e, sigma = %.2e, E = %.4e' %(ampl_potential, sigma_pot, functional_energy))
    plt.savefig('figures/' + fig_dir + '1D_cut_gauss', bbox_inches='tight')

    # Try iteration to see the best fitting gaussian
    fig, axes = plt.subplots(ncols=2, figsize=(10, 7))
    axes[0].plot(x, potential_target[int(n_points / 2), :], label='True potential')
    axes[1].plot(x, electric_field_norm_target[int(n_points / 2), :], label='True E_field')

    ampl_potential = 150
    pow_pot = 2
    potential = cosine_hill(X, Y, ampl_potential, x0, L, pow_pot)
    potential[:, 0], potential[0, :], potential[:, -1], potential[-1, :] = 0, 0, 0 ,0
    electric_field = grad(potential, dx, dy, n_points, n_points)
    electric_field_norm = np.sqrt(electric_field[0]**2 + electric_field[1]**2)
    functional_energy = func_energy(potential, electric_field, physical_rhs, voln)
    print('A = %.2e, sigma = %.2e - energy = %.4e' % (ampl_potential, sigma_pot, functional_energy))
    axes[0].plot(x, potential[int(n_points / 2), :])
    axes[0].legend()
    axes[1].plot(x, electric_field_norm[int(n_points / 2), :])
    axes[1].legend()
    plt.suptitle('A = %.2e, pow = %.2e, E = %.4e' %(ampl_potential, pow_pot, functional_energy))
    plt.savefig('figures/' + fig_dir + '1D_cut_coshill', bbox_inches='tight')

    # # Map the 2D energy functional
    # ampl_potential_range = np.linspace(160, 180, 21)
    # sigma_pot_range = np.linspace(1e-4, 3e-3, 41)   
    # count = 0
    # list_energy = []
    # for ampl_potential in ampl_potential_range:
    #     for sigma_pot in sigma_pot_range:
    #         potential = gaussian(X, Y, ampl_potential, x0, y0, sigma_pot, sigma_pot)
    #         potential[:, 0], potential[0, :], potential[:, -1], potential[-1, :] = 0, 0, 0 ,0
    #         electric_field = grad(potential, dx, dy, n_points, n_points)
    #         functional_energy = func_energy(potential, electric_field, physical_rhs, voln)
    #         list_energy.append([ampl_potential, sigma_pot, functional_energy])
    #         count += 1
    # energy_2D = np.array(list_energy)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(energy_2D[:, 0], energy_2D[:, 1], energy_2D[:, 2])
    # ax.set_xlabel('Amplitude')
    # ax.set_ylabel('Sigma')
    # ax.set_zlabel('Energy')
    # plt.savefig('figures/' + fig_dir + '2D_energy', bbox_inches='tight')
    # plt.show()

    # #Conversion to torch tensors
    # physical_rhs = torch.from_numpy(physical_rhs)
    # X, Y, voln = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(voln)

    # ampl_potential_range = np.linspace(160, 170, 11)
    # sigma_pot_range = np.linspace(1e-3, 5e-3, 20)
    # ampl_potential = 160
    # sigma_pot = 1e-3
    # v = torch.tensor([ampl_potential, sigma_pot], requires_grad=True)
    # potential = torch_gaussian(X, Y, v)
    # potential[:, 0], potential[0, :], potential[:, -1], potential[-1, :] = 0, 0, 0 ,0
    # electric_field = optorch.grad(potential, dx, dy, n_points, n_points)
    # functional_energy = func_energy_torch(potential, electric_field, physical_rhs, voln)
    # functional_energy.backward()
    # print(potential.grad_fn)
    # print(electric_field.grad_fn)
    # print(functional_energy.grad_fn)
    # print(v.grad)
    # ax.plot(x, potential[int(n_points / 2), :], label='A = %.2e, sigma = %.2e' %(ampl_potential, sigma_pot))
    # print('A = %.2e, sigma = %.2e I(phi) = %.4e' % (ampl_potential, sigma_pot, functional_energy))

    # ax.legend()
    # plt.savefig('figures/' + fig_dir + '1D_potential', bbox_inches='tight')