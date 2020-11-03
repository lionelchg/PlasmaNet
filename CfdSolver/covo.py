########################################################################################################################
#                                                                                                                      #
#                                         Drift-diffusion fluid plasma solver                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import yaml
import copy

from boundary import outlet_x, outlet_y, perio_x, perio_y, full_perio
from metric import compute_voln
from operators import grad
from plot import plot_scalar
from scheme import compute_flux


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


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
    geom = config['params']['geom']
    if geom == 'xr':
        R_nodes = copy.deepcopy(Y)
        R_nodes[0] = dy / 4
        voln = compute_voln(X, dx, dy) * R_nodes
    else:
        voln = compute_voln(X, dx, dy)
    sij = np.array([dy / 2, dx / 2])

    # Boundary conditions
    BC = config['BC']

    # Convection speed
    a = np.zeros((2, nny, nnx))
    a[0, :, :] = config['transport']['convection_x']
    a[1, :, :] = config['transport']['convection_y']
    norm_a = np.sqrt(a[0, :, :] ** 2 + a[1, :, :] ** 2)
    max_speed = np.max(norm_a)

    # Diffusion coefficient
    D = np.zeros_like(X)
    D = config['transport']['diffusion']
    max_D = np.max(D)

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
    fourier = config['params']['fourier']
    dt = min(cfl * dx / max_speed, fourier * dx ** 2 / max_D)
    dtsum = 0

    # Number of iterations
    nit = config['params']['nit']

    # Scalar and Residual declaration
    u, res = np.zeros_like(X), np.zeros_like(X)

    # Gaussian initialization
    u = gaussian(X, Y, 1, 0.5, 0.0, 1e-1, 1e-1)

    # Print header to sum up the parameters
    if verbose:
        print(f'Number of nodes: nnx = {nnx:d} -- nny = {nny:d}')
        print(f'Bounding box: ({xmin:.1e}, {ymin:.1e}), ({xmax:.1e}, {ymax:.1e})')
        print(f'Transport: a = {max_speed:.2e} -- D = {max_D:.2e}')
        print(f'dx = {dx:.2e} -- dy = {dy:.2e} -- CFL = {cfl:.2e} -- Fourier = {fourier:.2e} -- Timestep = {dt:.2e}')
        print('------------------------------------')
        print('Start of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    # Iterations
    for it in range(1, nit + 1):
        dtsum += dt
        # Calculation of diffusive flux
        diff_flux = D * grad(u, dx, dy, nnx, nny)
        # Update of the residual to zero
        res[:] = 0
        # Loop on the cells to compute the interior flux and update residuals
        if geom == 'xy':
            compute_flux(res, a, u, diff_flux, sij, ncx, ncy)
        elif geom == 'xr':
            compute_flux(res, a, u, diff_flux, sij, ncx, ncy, r=y)


        # Boundary conditions
        if BC == 'full_perio':
            full_perio(res)
        elif BC == 'perio_x':
            perio_x(res)
            outlet_y(res, a, u, diff_flux, dx, 0)
            outlet_y(res, a, u, diff_flux, dx, -1)
        elif BC == 'perio_y':
            perio_y(res)
            outlet_x(res, a, u, diff_flux, dy, 0)
            outlet_x(res, a, u, diff_flux, dy, -1)
        elif BC == 'full_out':
            if geom == 'xy':
                outlet_y(res, a, u, diff_flux, dx, 0)
                outlet_y(res, a, u, diff_flux, dx, -1)
                outlet_x(res, a, u, diff_flux, dy, 0)
                outlet_x(res, a, u, diff_flux, dy, -1)
            elif geom == 'xr':
                outlet_y(res, a, u, diff_flux, dx, -1, r=np.max(Y))
                outlet_x(res, a, u, diff_flux, dy, 0, r=Y)
                outlet_x(res, a, u, diff_flux, dy, -1, r=Y)

        
        u = u - res * dt / voln

        if verbose and (it % period == 0 or it == nit):
            print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, dt, dtsum, width=14))

        if save_type == 'iteration':
            if it % period == 0 or it == nit:
                plot_scalar(X, Y, u, res / voln, dtsum, number, fig_dir, geom=geom)
                number += 1
        elif save_type == 'time':
            if np.abs(dtsum - number * period) < 0.1 * dt or it == nit:
                plot_scalar(X, Y, u, res / voln, dtsum, number, fig_dir, geom=geom)
                number += 1
        elif save_type == 'none':
            pass


if __name__ == '__main__':

    with open('config.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)