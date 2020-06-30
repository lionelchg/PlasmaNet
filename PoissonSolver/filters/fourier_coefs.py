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
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_2D
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff

fig_dir = 'figures/rhs_2D_fourier/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def mode(X, Y, Lx, Ly, N, M):
    return np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly)

def sum_series(X, Y, Lx, Ly, coefs, N, M):
    series = np.zeros_like(X)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            series += coefs[n - 1, m - 1] * np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly)
    return series

def pot_series(X, Y, Lx, Ly, coefs, N, M):
    series = np.zeros_like(X)
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            series += coefs[n - 1, m - 1] * np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly) / ((n * np.pi / Lx)**2 + (m * np.pi / Ly)**2)
    return series

def plot_fourier(fx, fy, ampl, figname):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(fx, fy, ampl, 200, cmap='Blues')
    ax.set_title('Fourier transform')
    ax.set_xlabel('$f_x$ [m$^{-1}$]')
    ax.set_ylabel('$f_y$ [m$^{-1}$]')
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_aspect('equal')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def twod_filter(input, n_points, Lx, cutoff_y, cutoff_x, figname):
    """ Filters the signal through direct and reverse FFTs with cutoff frequencies. """
    transf = np.fft.fft2(input)
    freq = np.fft.fftfreq(n_points, Lx / n_points)
    Freq = np.fft.fftshift(np.fft.fftfreq(n_points, Lx / n_points))

    plot_fourier(Freq, Freq, np.abs(np.fft.fftshift(transf)), figname + '_ft')

    for j in range(n_points):
        for i in range(n_points):
            if np.abs(freq[j]) > cutoff_y or np.abs(freq[i]) > cutoff_x:
                transf[j, i] = 0

    plot_fourier(Freq, Freq, np.abs(np.fft.fftshift(transf)), figname + '_ft_lp')

    output = np.fft.ifft2(transf)
    return np.real(output)

def mirror(field, n_points):
    field_mirror = np.zeros((2 * n_points - 1, 2 * n_points - 1))
    field_mirror[n_points - 1:, n_points - 1:] = field
    field_mirror[:n_points - 1, :n_points - 1] = field[-1:0:-1, -1:0:-1]
    field_mirror[n_points - 1:, :n_points - 1] = - field[:, -1:0:-1]
    field_mirror[:n_points - 1, n_points - 1:] = - field[-1:0:-1, :]

    return field_mirror

if __name__ == '__main__':
    n_points = 101
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)

    A = laplace_square_matrix(n_points)

    potential = np.zeros((n_points, n_points))
    physical_rhs = np.zeros((n_points, n_points))

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.5e-2, 0.5e-2
    rhs = np.zeros(n_points**2)

    # interior rhs
    N, M = 15, 15
    np.random.seed(30)
    random_array = np.random.random((N, M))
    print(random_array)
    rhs_coefs = ni0 * (2 * random_array - 1)
    physical_rhs = sum_series(X, Y, Lx, Ly, rhs_coefs, N, M)
    physical_rhs = physical_rhs.reshape(n_points**2)
    rhs = - physical_rhs * dx ** 2

    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)

    # Solving the sparse linear system
    potential = spsolve(A, rhs).reshape(n_points, n_points)
    physical_rhs = physical_rhs.reshape(n_points, n_points)
    E_field = - grad(potential, dx, dy, n_points, n_points)
    interior_diff = lapl_diff(potential, physical_rhs, dx, dy, n_points, n_points)

    casename = 'solver'
    figname = fig_dir + casename
    # Plots
    plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Fourier random', figname)

    potential_th = pot_series(X, Y, Lx, Ly, rhs_coefs, N, M)
    E_field_th = - grad(potential_th, dx, dy, n_points, n_points)
    interior_diff_th = lapl_diff(potential_th, physical_rhs, dx, dy, n_points, n_points)

    casename = 'th'
    figname = fig_dir + casename
    # Plots
    plot_set_2D(X, Y, physical_rhs, potential_th, E_field_th, 'Fourier random', figname)

    Nlow, Mlow = 3, 3
    potential_th = pot_series(X, Y, Lx, Ly, rhs_coefs[:Nlow, :Mlow], Nlow, Mlow)
    E_field_th = - grad(potential_th, dx, dy, n_points, n_points)
    physical_rhs = sum_series(X, Y, Lx, Ly, rhs_coefs[:Nlow, :Mlow], Nlow, Mlow)
    interior_diff_th = lapl_diff(potential_th, physical_rhs, dx, dy, n_points, n_points)

    casename = 'th_low_%d' % Nlow
    figname = fig_dir + casename
    # Plots
    plot_set_2D(X, Y, physical_rhs, potential_th, E_field_th, 'Fourier random', figname)

    physical_rhs = sum_series(X, Y, Lx, Ly, rhs_coefs, N, M)
    potential_th = pot_series(X, Y, Lx, Ly, rhs_coefs, N, M)

    fx_cut, fy_cut = Nlow / 2 / Lx, Mlow / 2 / Lx

    # Non-mirror
    rhs_filt = twod_filter(physical_rhs, n_points, Lx, fx_cut, fy_cut, fig_dir + 'rhs_non_mirror')
    potential_filt = twod_filter(potential_th, n_points, Lx, fx_cut, fy_cut, fig_dir + 'potential_non_mirror')
    E_field_filt = - grad(potential_filt, dx, dy, n_points, n_points)
    # Plot
    casename = 'filtered_solution'
    figname = fig_dir + casename
    plot_set_2D(X, Y, rhs_filt, potential_filt, E_field_filt, 'Fourier random', figname)

    # Mirrored solution
    casename = 'mirror_solution'
    figname = fig_dir + casename

    physical_rhs_mirror = mirror(physical_rhs, n_points)
    potential_mirror = mirror(potential_th, n_points)
    E_mirror = - grad(potential_mirror, dx, dy, 2 * n_points - 1, 2 * n_points - 1)

    x, y = np.linspace(- xmax, xmax, 2 * n_points - 1), np.linspace(- ymax, ymax, 2 * n_points - 1)
    X, Y = np.meshgrid(x, y)
    
    plot_set_2D(X, Y, physical_rhs_mirror, potential_mirror, E_mirror, 'Fourier random', figname)

    # # Non-mirror
    casename = 'filt_mirror_solution'
    figname = fig_dir + casename
    rhs_filt = twod_filter(physical_rhs_mirror, 2 * n_points - 1, 2 * Lx, fx_cut, fy_cut, fig_dir + 'rhs_mirror')
    potential_filt = twod_filter(potential_mirror, 2 * n_points - 1, 2 * Lx, fx_cut, fy_cut, fig_dir + 'potential_mirror')
    E_field_filt = - grad(potential_filt, dx, dy, 2 * n_points - 1, 2 * n_points - 1)

    plot_set_2D(X, Y, rhs_filt, potential_filt, E_field_filt, 'Fourier random', figname)

