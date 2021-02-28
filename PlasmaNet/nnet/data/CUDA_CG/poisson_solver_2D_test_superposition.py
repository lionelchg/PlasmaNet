########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using CUDA                                              #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                    Posterior modifications by Ekhi Ajuria, 23.03.20                                  # 
#                                                                                                                      #
########################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.constants as co
from operators import L1_error, L2_error
from plot import plot_fig, plot_ax
from poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc
from scipy.sparse.linalg import spsolve
from solve_linear_sys import solveLinearSystemCG
import scipy
from scipy.sparse import *
from scipy import *

def print_error(computed, analytical, name):
    print(
        '%s L1 error = %.2e - L2 error = %.2e' % (name, L1_error(computed, analytical), L2_error(computed, analytical)))


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


if __name__ == '__main__':
    n_points = 11
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    #######################
    # rhs, zero dirichlet #
    #######################

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.8e-2, 0.8e-2
    rhs = np.zeros(n_points ** 2)

    # ZERO DIRICHLET
    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential_0 = torch.zeros((n_points,n_points))
    rhs_in = torch.FloatTensor(rhs)

    A_val_l = csr_matrix(A).data
    I_A_l = csr_matrix(A).indptr
    J_A_l = csr_matrix(A).indices

    I_A = torch.IntTensor(I_A_l)
    A_val = torch.FloatTensor(A_val_l)
    J_A = torch.IntTensor(J_A_l)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    tol  = solveLinearSystemCG(potential_0,rhs_in, A_val, I_A, J_A)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded! 
    elapsed_time_ms = start_event.elapsed_time(end_event)

    print("elapsed time CUDA: ", elapsed_time_ms)

    start_event_1 = torch.cuda.Event(enable_timing=True)
    end_event_1 = torch.cuda.Event(enable_timing=True)
    start_event_1.record()
  
    potential_0 = spsolve(A, rhs).reshape(n_points, n_points)

    end_event_1.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded! 
    elapsed_time_ms_1 = start_event_1.elapsed_time(end_event_1)

    print("elapsed time numpy: ", elapsed_time_ms_1)

    physical_rhs_0 = physical_rhs.reshape(n_points, n_points)
    plot_fig(X, Y, potential_0, physical_rhs_0, name='superposition/potential_', nit=0)

    ##################
    # dirichlet only #
    ##################

    # interior rhs
    physical_rhs = np.zeros(n_points ** 2)
    rhs = - physical_rhs * dx ** 2

    # dirichlet boundary conditions
    V = 1000
    ones_bc = np.ones(n_points)
    linear_bc = np.linspace(0, V, n_points)
    down = linear_bc
    up = linear_bc
    left = zeros_bc
    right = V * ones_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)

    print('rhs ', rhs)

    rhs_in = torch.FloatTensor(rhs)
    rhs_test = torch.zeros_like(rhs_in)
    for i in range(11):
        rhs_test[11*i] = i
        rhs_test[11*i+10] = i
    rhs_test[-11:]=10

    print("rhs test", rhs_test)

    potential_dirichlet = torch.zeros((n_points,n_points))

    tol  = solveLinearSystemCG(potential_dirichlet,rhs_test, A_val, I_A, J_A)
    #potential_dirichlet = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs_dirichlet = physical_rhs.reshape(n_points, n_points)
    plot_fig(X, Y, potential_dirichlet, physical_rhs_dirichlet, name='superposition/potential_', nit=1)

    ################
    # full problem #
    ################

    # interior rhs
    physical_rhs = gaussian(X.reshape(-1), Y.reshape(-1), ni0, x0, y0, sigma_x, sigma_y) * co.e / co.epsilon_0
    rhs = - physical_rhs * dx ** 2

    # dirichlet boundary conditions
    V = 1000
    ones_bc = np.ones(n_points)
    linear_bc = np.linspace(0, V, n_points)
    down = linear_bc
    up = linear_bc
    left = zeros_bc
    right = V * ones_bc

    dirichlet_bc(rhs, n_points, down, up, left, right)
    potential = torch.zeros((n_points,n_points))
    rhs_in = torch.FloatTensor(rhs)
    tol  = solveLinearSystemCG(potential,rhs_in, A_val, I_A, J_A)
    #potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    plot_fig(X, Y, potential, physical_rhs, name='superposition/potential_', nit=2)

    potential_super = potential_0 + potential_dirichlet
    plot_fig(X, Y, potential_super, physical_rhs.reshape(n_points, n_points), name='superposition/potential_', nit=3)

    print_error(potential_super.numpy(), potential.numpy(), "Error from superposition")

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 15))
    levels = plot_ax(fig, axes[0], X, Y, potential_0, physical_rhs_0, npot=1)
    plot_ax(fig, axes[1], X, Y, potential_dirichlet, physical_rhs_dirichlet, levels=levels, npot=2)
    plot_ax(fig, axes[2], X, Y, potential, physical_rhs)
    plt.savefig('../../../datasets/Ekhi_test/figures/superposition/superposition', bbox_inches='tight')
