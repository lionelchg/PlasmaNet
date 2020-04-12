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
import os
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

# Imports from poissonsolver
from poissonsolver.operators import lapl, grad
import poissonsolver.operators_torch as optorch
from poissonsolver.plot import plot_fig, plot_fig_scalar, plot_vector_arrow, plot_fig_list, plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff, compute_voln, func_energy, func_energy_torch
import poissonsolver.plot_comp as plot_comp

fig_dir = 'figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def torch_gaussian(x, y, v):
    return v[0] * torch.exp(-((x - 5e-3) / v[1]) ** 2 - ((y - 5e-3) / v[1]) ** 2)

def cosine_hill(x, y, amplitude, x0, L, power):
    r = np.sqrt((x - x0)**2 + (y - x0)**2)
    return amplitude * (np.cos(2 * np.pi / L * r))

def torch_cosine_hill(x, y, x0, L):
    r = torch.sqrt((x - x0)**2 + (y - x0)**2)
    return torch.cos(2 * np.pi / L * r)

def compute_values(ampl_potential, sigma_pot, print_bool=False):
    potential = gaussian(X, Y, ampl_potential, x0, y0, sigma_pot, sigma_pot) * cosine_hill(X, Y, 1, x0, 2 * L, 1)
    E_field = grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    functional_energy = func_energy(potential, E_field, physical_rhs, voln) - functional_energy_target
    points_loss = np.sum((potential - potential_target)**2) / n_points**2
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)
    interior_diff = lapl_diff(potential, physical_rhs, dx, dy, n_points, n_points)
    lapl_loss = np.sum(interior_diff**2) / n_points**2
    elec_loss = np.sum((E_field_norm - E_field_norm_target)**2) / n_points**2

    info_test = 'A = %.2e, sigma = %.2e - energy = %.4e - points_loss = %.4e - lapl_loss = %.4e - elec_loss = %.4e' \
            % (ampl_potential, sigma_pot, functional_energy, points_loss, lapl_loss, elec_loss)
    if print_bool: print(info_test)

    return potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, lapl_loss, elec_loss, info_test

def plot_test(ampl_potential, sigma_pot, test_name):
    # Single test
    potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, lapl_loss, elec_loss, info_test = \
                compute_values(ampl_potential, sigma_pot, print_bool=True)

    # 1D plots
    plot_comp.plot_set_1D(x, potential_target, E_field_norm_target, lapl_target, potential, E_field_norm, lapl_pot, n_points, info_test, fig_dir + test_name + 'comp_1D')
    # 2D plots
    plot_comp.plot_set_2D(X, Y, physical_rhs, potential, potential_target, E_field, E_field_target, info_test, fig_dir + test_name + 'comp_2D')

if __name__ == '__main__':
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
    E_field_target = grad(potential_target, dx, dy, n_points, n_points)
    E_field_norm_target = np.sqrt(E_field_target[0]**2 + E_field_target[1]**2)
    lapl_target = lapl(potential_target, dx, dy, n_points, n_points)
    field_energy_target = co.epsilon_0 / 2 * (E_field_target[0]**2 + E_field_target[1]**2)
    potential_energy_target = physical_rhs * co.epsilon_0 * potential_target
    interior_diff_target = lapl_diff(potential_target, physical_rhs, dx, dy, n_points, n_points)
    functional_energy_target = func_energy(potential_target, E_field_target, physical_rhs, voln)

    # Plots of the target
    plot_set_2D(X, Y, physical_rhs, potential_target, E_field_target, 'Target values', fig_dir + 'target')

    # Plot of one test
    plot_test(146, 3.5e-3, 'test_1_')

    # Map the 2D energy functional
    ampl_potential_range = np.linspace(130, 180, 51)
    sigma_pot_range = np.linspace(1e-3, 5e-3, 41)  
    count = 0
    list_loss = []
    for ampl_potential in ampl_potential_range:
        for sigma_pot in sigma_pot_range:
            potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, lapl_loss, elec_loss, info_test = \
                compute_values(ampl_potential, sigma_pot)
            list_loss.append([ampl_potential, sigma_pot, functional_energy, points_loss, lapl_loss, elec_loss])
            count += 1
    losses_2D = np.array(list_loss)

    loss_list_name = ['Energy func', 'Points', 'Lapl', 'Elec']
    index_min_list = [np.argmin(losses_2D[:, 2]), np.argmin(losses_2D[:, 3]), np.argmin(losses_2D[:, 4]), np.argmin(losses_2D[:, 5])]
    for i in range(4):
        loss = loss_list_name[i]
        index_min = index_min_list[i]
        print('%s A = %.2e, sigma = %.2e - energy = %.4e - points_loss = %.4e - lapl_loss = %.4e - elec_loss = %.4e' 
            % (loss, losses_2D[index_min, 0], losses_2D[index_min, 1], losses_2D[index_min, 2], losses_2D[index_min, 3], 
                losses_2D[index_min, 4], losses_2D[index_min, 5]))
        plot_test(losses_2D[index_min, 0], losses_2D[index_min, 1], 'test_%d_' % i)
    
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(221, projection='3d')

    ax.scatter(losses_2D[:, 0], losses_2D[:, 1], losses_2D[:, 2])
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Energy')
    ax.set_title('Energy functional')

    ax = fig.add_subplot(222, projection='3d')

    ax.scatter(losses_2D[:, 0], losses_2D[:, 1], losses_2D[:, 3])
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Points Loss')
    ax.set_title('Points Loss')

    ax = fig.add_subplot(223, projection='3d')

    ax.scatter(losses_2D[:, 0], losses_2D[:, 1], losses_2D[:, 4])
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Lapl Loss')
    ax.set_title('Laplacian Loss')

    ax = fig.add_subplot(224, projection='3d')

    ax.scatter(losses_2D[:, 0], losses_2D[:, 1], losses_2D[:, 5])
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Sigma')
    ax.set_zlabel('Elec Loss')
    ax.set_title('Electric Loss')

    plt.savefig(fig_dir + '2D_losses', bbox_inches='tight')

    # #Conversion to torch tensors
    # physical_rhs = torch.from_numpy(physical_rhs)
    # X, Y, voln = torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(voln)

    # ampl_potential = 160
    # sigma_pot = 2e-3
    # eta = 1e-7
    # for i in range(5):
    #     print('Step %d' % (i + 1))
    #     v = torch.tensor([ampl_potential, sigma_pot], requires_grad=True)
    #     potential = torch_gaussian(X, Y, v) * torch_cosine_hill(X, Y, x0, 2 * L)
    #     E_field = optorch.grad(potential, dx, dy, n_points, n_points)
    #     functional_energy = func_energy_torch(potential, E_field, physical_rhs, voln)
    #     functional_energy.backward()
    #     print(potential.grad_fn)
    #     print(E_field.grad_fn)
    #     print(functional_energy.grad_fn)
    #     print('A = %.2e, sigma = %.2e I(phi) = %.4e' % (ampl_potential, sigma_pot, functional_energy))
    #     ampl_potential -= eta * v.grad.data[0]
    #     sigma_pot -= eta * v.grad.data[1]

