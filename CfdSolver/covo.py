########################################################################################################################
#                                                                                                                      #
#                               Convective vortex for validation of Euler integration                                  #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import yaml
import copy
import scipy.constants as co
import matplotlib.pyplot as plt

from boundary import impose_bc_euler
from metric import compute_voln
from operators import grad
from plot import plot_euler, plot_ax_scalar
from euler import compute_flux, compute_res

def print_init(nnx, nny, xmin, ymin, xmax, ymax, dx, dy, cfl, dt):
    # Print header to sum up the parameters
    print(f'Number of nodes: nnx = {nnx:d} -- nny = {nny:d}')
    print(f'Bounding box: ({xmin:.1e}, {ymin:.1e}), ({xmax:.1e}, {ymax:.1e})')
    print(f'dx = {dx:.2e} -- dy = {dy:.2e} -- CFL = {cfl:.2e} -- Timestep = {dt:.2e}')
    print('------------------------------------')
    print('Start of simulation')
    print('------------------------------------')
    print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def covo(x, y, x0, y0, u0, v0, rho0, p0, T0, alpha, K, gamma, r, t, U):
    xbar = x - x0 - u0 * t
    ybar = y - y0 - v0 * t
    rbar = np.sqrt(xbar**2 + ybar**2)
    u = u0 - K / (2 * np.pi) * ybar * np.exp(alpha * (1 - rbar**2) / 2)
    v = v0 + K / (2 * np.pi) * xbar * np.exp(alpha * (1 - rbar**2) / 2)
    T = T0 - K**2 * (gamma - 1) / (8 * alpha * np.pi**2 * gamma * r) * np.exp(alpha * (1 - rbar**2))
    rho = rho0 * (T / T0)**(1 / (gamma - 1))
    p = p0 * (T / T0)**(gamma / (gamma - 1))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v
    U[3] = rho / 2 * (u**2 + v**2) + p / (gamma - 1)

def main(config):
    """ Main function containing initialisation, temporal loop and outputs. Takes a config dict as input. """

    # Mesh properties
    nnx, nny = config['mesh']['nnx'], config['mesh']['nny']
    ncx, ncy = nnx - 1, nny - 1  # Number of cells
    xmin, xmax = config['mesh']['xmin'], config['mesh']['xmax']
    ymin, ymax = config['mesh']['ymin'], config['mesh']['ymax']
    Lx, Ly = xmax - xmin, ymax - ymin
    dx = (xmax - xmin) / ncx
    dy = (ymax - ymin) / ncy
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)

    # Grid construction
    X, Y = np.meshgrid(x, y)
    voln = compute_voln(X, dx, dy)
    volc = dx * dy
    snc = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])

    # Boundary conditions
    BC = config['BC']

    # Creation of the figures directory and numbering of outputs
    fig_dir = 'figures/' + config['casename']
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    number = 1
    save_type = config['output']['save']
    verbose = config['output']['verbose']
    period = config['output']['period']

    # Timestep calculation through cfl and fourier
    cfl = config['params']['cfl']
    dtsum = 0

    # Number of iterations
    nit = config['params']['nit']

    # Scalar and Residual declaration - Mixture parameters
    gamma = 7 / 5
    neqs = 4
    ndim = 2
    nvert = 4
    Wair = 0.029 # kg/mol
    rair = co.R / Wair
    U = np.zeros((neqs, nny, nnx))
    U_c = np.zeros((neqs, ncy, ncx))
    F = np.zeros((neqs, ndim, nny, nnx))
    res = np.zeros_like(U)
    res_c = np.zeros((neqs, ncy, ncx))
    press, Tgas = np.zeros_like(X), np.zeros_like(X)

    # Convective vortex parameters and initialization
    x0, y0 = 10, 10
    u0, v0 = 2, 2
    rho0, p0 = 1, 1 / gamma
    alpha, K = 1, 5
    a0 = np.sqrt(gamma * p0 / rho0)
    T0 = 1 / gamma / rair
    time = 0
    covo(X, Y, x0, y0, u0, v0, rho0, p0, T0, alpha, K, gamma, rair, time, U)
    dt = cfl * dx / a0
    # Print header to sum up the parameters
    if verbose:
        print_init(nnx, nny, xmin, ymin, xmax, ymax, dx, dy, cfl, dt)
    

    # Iterations
    for it in range(1, nit + 1):
        # Update of the residual to zero
        res[:], res_c[:] = 0, 0
        # Loop on the cells to compute the interior flux and update residuals

        compute_flux(U, gamma, rair, F, press, Tgas)

        compute_res(U, F, press, dt, snc, ncx, ncy, gamma, ndim, nvert, res, res_c, U_c)

        impose_bc_euler(BC, res)
        
        U = U - res / voln

        dtsum += dt

        if verbose and (it % period == 0 or it == nit):
            print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, dt, dtsum, width=14))

        if save_type == 'iteration':
            if it % period == 0 or it == nit:
                plot_euler(X, Y, U, gamma, u0, v0, dtsum, fig_dir + f'2D_{it:04d}')
                number += 1
        elif save_type == 'time':
            if np.abs(dtsum - number * period) < 0.1 * dt or it == nit:
                plot_euler(X, Y, U, gamma, u0, v0, dtsum, fig_dir + f'2D_{it:04d}')
                number += 1
        elif save_type == 'none':
            pass


if __name__ == '__main__':

    with open('covo.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)
