########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
from .operators import errors
from .plot import plot_fig
from .poisson_setup_2D_FD import laplace_square_matrix, dirichlet_bc
from scipy.sparse.linalg import spsolve


def run_poisson(n_points, kx=2, ky=2, plot=False):
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
    rhs = np.zeros(n_points ** 2)

    # interior rhs
    physical_rhs = (2 * kx * np.pi / Lx) ** 2 * np.cos(kx * 2 * np.pi * X.reshape(-1) / Lx) \
                   + (2 * ky * np.pi / Ly) ** 2 * np.cos(ky * 2 * np.pi * Y.reshape(-1) / Ly)
    rhs = physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    tgv_boundary_x = - (np.cos(kx * 2 * np.pi * x / Lx) + 1)
    tgv_boundary_y = - (np.cos(ky * 2 * np.pi * y / Ly) + 1)
    dirichlet_bc(rhs, n_points, tgv_boundary_x, tgv_boundary_x, tgv_boundary_y, tgv_boundary_y)

    # Solving the sparse linear system
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    analytical_pot = - (np.cos(kx * 2 * np.pi * X / Lx) + np.cos(ky * 2 * np.pi * Y / Ly))

    if plot:
        plot_fig(X, Y, potential, physical_rhs, name='tgv/n_%d_kx_%d_ky_%d' % (n_points, kx, ky), nit=0)

    return errors(potential, analytical_pot, dx * dy, Lx * Ly)


if __name__ == '__main__':

    list_n_points = [51, 101, 201, 501]
    list_L1, list_L2, list_Linf, list_dx = [], [], [], []
    for n_points in list_n_points:
        print('n_points = %d' % n_points)
        if n_points == 101:
            plot = True
        else:
            plot = False
        L1, L2, Linf = run_poisson(n_points, plot=plot)
        list_dx.append(1 / n_points)
        list_L1.append(L1)
        list_L2.append(L2)
        list_Linf.append(Linf)

    fig, ax = plt.subplots()

    ax.plot(list_dx, list_L1, label='L1 norm')
    ax.plot(list_dx, list_L2, label='L2 norm')
    ax.plot(list_dx, list_Linf, label='Linf norm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(r'$\Delta x$ [m]', size=12)
    ax.set_ylabel(r'$\epsilon$ [V]', size=12)
    plt.suptitle('Convergence error for the TGV $k_x = k_y = 2$')
    plt.savefig('figures/tgv/tgv_convergence_1')

    list_n_points = [51, 101, 201, 401, 501]
    list_L1, list_L2, list_Linf, list_dx = [], [], [], []
    for n_points in list_n_points:
        print('n_points = %d' % n_points)
        if n_points == 101:
            plot = True
        else:
            plot = False
        L1, L2, Linf = run_poisson(n_points, kx=8, ky=8, plot=plot)
        list_dx.append(1 / n_points)
        list_L1.append(L1)
        list_L2.append(L2)
        list_Linf.append(Linf)

    fig, ax = plt.subplots()

    ax.plot(list_dx, list_L1, label='L1 norm')
    ax.plot(list_dx, list_L2, label='L2 norm')
    ax.plot(list_dx, list_Linf, label='Linf norm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel(r'$\Delta x$ [m]', size=12)
    ax.set_ylabel(r'$\epsilon$ [V]', size=12)
    plt.suptitle('Convergence error for the TGV $k_x = k_y = 8$')
    plt.savefig('figures/tgv/tgv_convergence_2')

    # error_lapl = abs(lapl(potential, dx, dy, n_points, n_points) - physical_rhs)
    # error_lapl[0, :] = 0
    # error_lapl[-1, :] = 0
    # error_lapl[:, 0] = 0
    # error_lapl[:, -1] = 0
    # error_L1 = abs(potential - analytical_pot)

    # print_error(potential, analytical_pot, dx*dy, Lx*Ly, "Analytical - Computed")

    # plot_fig(X, Y, potential, physical_rhs, name='tgv/computed_', nit=1)
    # plot_fig(X, Y, analytical_pot, physical_rhs, name='tgv/analytical_', nit=1)
    # plot_fig_scalar(X, Y, error_lapl, 'Absolute difference Laplacian', 'tgv/abs_diff_lapl')
    # plot_fig_scalar(X, Y, error_L1, 'L1 diff', 'tgv/abs_diff_L1')
