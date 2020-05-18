########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import sys
import numpy as np
import scipy.constants as co
from scipy.sparse.linalg import spsolve
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from poissonsolver.operators import dv, dv2
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff, compute_voln

mpl.rcParams['lines.linewidth'] = 2


def L1_error(y, y_th, n):
    return np.sum(abs(y - y_th)) / n

def L2_error(y, y_th, n):
    return np.sqrt(np.sum((y - y_th)**2)) / n

def Linf_error(y, y_th):
    return np.max(np.abs(y - y_th))

def sup_error(n, Lx, alpha):
    return Lx**2 / np.pi**2 / (1 + alpha) / n**(alpha + 1)

def gaussian(x, amplitude, x0, sigma_x):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2)

def triangle(x, ampl, x0, sigma):
    return ampl * np.maximum(0, 1 - np.abs((x - x0) / sigma))

def step(x, ampl, x0, sigma):
    return ampl * (np.sign(1 - np.abs(2 * (x - x0) / sigma)) + 1) / 2

def integral_term(x, Lx, voln, rhs, n):
    return 2 / Lx * np.sum(np.sin(n * np.pi * x / Lx) * rhs * voln)

def fourier_coef(x, Lx, voln, rhs, n):
    return integral_term(x, Lx, voln, rhs, n) / (n / Lx)**2 / np.pi**2

def series_term(x, Lx, voln, rhs, n):
    return fourier_coef(x, Lx, voln, rhs, n) * np.sin(n * np.pi * x / Lx)

def sum_series(x, Lx, voln, rhs, N):
    series = np.zeros_like(x)
    for n in range(1, N + 1):
        series += series_term(x, Lx, voln, rhs, n)
    return series

def ax_prop(ax, xlabel, ylabel):
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_potential(x, dx, pot, nx, figname):
    E_field = - dv(pot, x, dx)
    lapl = dv2(pot, x, dx)

    fig, axes = plt.subplots(ncols=3, figsize=(14, 5))
    axes[0].plot(x, pot)
    ax_prop(axes[0], '$x$ [m]', '$\\phi$ [V]')

    axes[1].plot(x, E_field)
    ax_prop(axes[1], '$x$ [m]', '$E$ [V/m]')

    axes[2].plot(x, - lapl)
    ax_prop(axes[2], '$x$ [m]', '$-\\Delta \\phi$ [V/m2]')

    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')


if __name__ == '__main__':

    fig_dir = 'figures/rhs_1D/%s_offcenter/' % sys.argv[1]

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    nx = 501
    xmin, xmax = 0, 0.01
    Lx = xmax - xmin
    dx = (xmax - xmin) / (nx - 1)
    x = np.linspace(xmin, xmax, nx)

    voln = np.ones_like(x) * dx
    voln[0], voln[-1] = dx / 2, dx / 2

    # creating the rhs
    ni0, sigma_x, x0 = 1e16, 2e-3, 0.2e-2

    # dictionnary of trial functions
    trial_function = dict([('triangle', triangle), ('step', step), ('gaussian', gaussian)])

    # interior rhs
    physical_rhs = trial_function[sys.argv[1]](x, ni0, x0, sigma_x) * co.e / co.epsilon_0

    # Analytical solution but with a quadrature formula for the Fourier coefficient
    # the converged solution is with N = 21
    ref_potential = sum_series(x, Lx, voln, physical_rhs, 31)
    # list_N = [1, 3, 5, 9, 15, 21]
    list_N = np.arange(1, 21, 2)
    errors = np.zeros((len(list_N), 3))
    for i, N in enumerate(list_N):
        potential_th = sum_series(x, Lx, voln, physical_rhs, N)
        casename = 'fourier_%02d' % N
        figname = fig_dir + casename
        plot_potential(x, dx, potential_th, nx, figname)
        errors[i, :] = np.array([L1_error(potential_th, ref_potential, nx), L2_error(potential_th, ref_potential, nx), 
                                Linf_error(potential_th, ref_potential)])

    # Plot of the modes
    nrange = np.arange(21)
    Coeff = np.zeros(nrange.shape)
    for index, order in enumerate(nrange[1:]):
            Coeff[index + 1] = fourier_coef(x, Lx, voln, physical_rhs, order)
            # print('%d %d %.2e' % (i, j, Coeff[j - 1, i - 1]))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(nrange, np.abs(Coeff), drawstyle='steps-mid')
    ax.grid(True)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('N')
    ax.set_title('Mode amplitudes')
    ax.set_yscale('log')
    ax.set_ylim([1e-2, 2e3])
    plt.tight_layout()
    plt.savefig(fig_dir + 'mode_amplitudes', bbox_inches='tight')

    # Plot of the error
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(list_N, errors[:, 0], label='L1')
    ax.plot(list_N, errors[:, 1], label='L2')
    ax.plot(list_N, errors[:, 2], label='Linf')
    for alpha in range(1, 3):
        ax.plot(list_N, ni0 * co.e / co.epsilon_0 * sup_error(list_N, Lx, alpha), label=r'sup $\alpha = %d$' % alpha)
    ax.legend()
    ax.grid(True)
    ax.set_ylabel(r'$L(\phi_{ref} - \phi)$')
    ax.set_xlabel('N')
    ax.set_title('Residuals')
    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1e3])
    plt.tight_layout()
    plt.savefig(fig_dir + 'residuals', bbox_inches='tight')