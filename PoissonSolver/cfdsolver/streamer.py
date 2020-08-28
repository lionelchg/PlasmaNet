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
import scipy.constants as co
import copy

from boundary import outlet_x, outlet_y, perio_x, perio_y, full_perio
from metric import compute_voln
from operators import grad
from plot import plot_scalar, plot_streamer
from scheme import compute_flux
from chemistry import morrow

from scipy.sparse.linalg import spsolve
from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_2D, plot_potential
from poissonsolver.linsystem import matrix_cart, matrix_axisym, dirichlet_bc_axi
from poissonsolver.postproc import lapl_diff


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
    R_nodes = copy.deepcopy(Y)
    R_nodes[0] = dy / 2
    voln = compute_voln(X, dx, dy) * R_nodes
    sij = np.array([dy / 2, dx / 2])

    # Convection speed
    a = np.zeros((2, nny, nnx))

    # Transport coefficients
    mu = np.zeros_like(X)
    D = np.zeros_like(X)

    # Creation of the figures directory and numbering of outputs
    fig_dir = 'figures/' + config['casename']
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    number = 1
    save_type = config['output']['save']
    verbose = config['output']['verbose']
    period = config['output']['period']

    # Timestep fixed
    dt = 5e-13
    dtsum = 0

    # Background electric field
    backE = float(config['poisson']['backE'])
    up = x * backE
    left = np.zeros_like(y)
    right = np.ones_like(y) * backE * xmax

    # Number of iterations
    nit = config['params']['nit']

    # Scalar and Residual declaration
    ne, rese = np.zeros_like(X), np.zeros_like(X)
    nionp, resp = np.zeros_like(X), np.zeros_like(X)
    nn, resn = np.zeros_like(X), np.zeros_like(X)

    # Gaussian initialization for the electrons and positive ions
    ne = gaussian(X, Y, 1e19, 2e-3, 0, 2e-4, 2e-4)
    nionp = gaussian(X, Y, 1e19, 2e-3, 0, 2e-4, 2e-4)

    # Print header to sum up the parameters
    if verbose:
        print(f'Number of nodes: nnx = {nnx:d} -- nny = {nny:d}')
        print(f'Bounding box: ({xmin:.1e}, {ymin:.1e}), ({xmax:.1e}, {ymax:.1e})')
        print(f'dx = {dx:.2e} -- dy = {dy:.2e} -- Timestep = {dt:.2e}')
        print('------------------------------------')
        print('Start of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    # Iterations
    for it in range(1, nit + 1):
        dtsum += dt

        # Solve the Poisson equation / Axisymmetric resolution
        physical_rhs = (nionp - ne - nn).reshape(-1) * co.e / co.epsilon_0
        A = matrix_axisym(dx, dy, nnx, nny, R_nodes)
        rhs = - physical_rhs
        dirichlet_bc_axi(rhs, nnx, nny, up, left, right)
        potential = spsolve(A, rhs).reshape(nny, nnx)
        E_field = - grad(potential, dx, dy, nnx, nny)
        physical_rhs = physical_rhs.reshape((nny, nnx))

        # Update of the residual to zero
        rese[:], resp[:], resn[:] = 0, 0, 0

        # Application of chemistry
        morrow(mu, D, E_field, ne, rese, nionp, resp, nn, resn, nnx, nny)

        # Convective and diffusive flux
        a = mu * E_field
        diff_flux = D * grad(ne, dx, dy, nnx, nny)

        # Loop on the cells to compute the interior flux and update residuals
        compute_flux(rese, a, ne, diff_flux, sij, ncx, ncy, r=y)

        # Boundary conditions
        outlet_y(rese, a, ne, diff_flux, dx, -1)
        outlet_x(rese, a, ne, diff_flux, dy, 0)
        outlet_x(rese, a, ne, diff_flux, dy, -1)

        ne = ne - rese * dt / voln
        nionp = nionp - resp * dt / voln
        nn = nn - resn * dt / voln

        if verbose and (it % period == 0 or it == nit):
            print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, dt, dtsum, width=14))

        if save_type == 'iteration':
            if it % period == 0 or it == nit:
                plot_streamer(X, Y, ne, rese, nionp, resp, nn, resn, dtsum, number, fig_dir)
                plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Poisson fields', fig_dir + 'EM_instant_%04d' % number, no_rhs=False)
                number += 1
        elif save_type == 'time':
            if np.abs(dtsum - number * period) < 0.1 * dt or it == nit:
                plot_streamer(X, Y, ne, rese, nionp, resp, nn, resn, dtsum, number, fig_dir)
                plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Poisson fields', fig_dir + 'EM_instant_%04d' % number, no_rhs=False)
                number += 1
        elif save_type == 'none':
            pass


if __name__ == '__main__':

    with open('config_streamer.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)
