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
import re

from boundary import outlet_x, outlet_y, perio_x, perio_y, full_perio
from metric import compute_voln
from operators import grad
from plot import plot_scalar, plot_streamer, plot_streamer_1D, plot_global
from scheme import compute_flux
from chemistry import morrow

from scipy.sparse.linalg import spsolve, isolve, cg, cgs
from poissonsolver.operators import lapl, grad
from poissonsolver.plot import plot_set_1D, plot_set_2D, plot_potential
from poissonsolver.linsystem import matrix_cart, matrix_axisym, dirichlet_bc_axi
from poissonsolver.postproc import lapl_diff

from photo import photo_axisym, lambda_j_two, A_j_two, lambda_j_three, A_j_three, plot_Sph_irate

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def plot_it(X, Y, ne, rese, nionp, resp, nn, resn, physical_rhs, potential, E_field, lapl_pot, voln, dtsum, number, fig_dir):
    plot_streamer(X, Y, ne, rese / voln, nionp, resp / voln, nn, resn / voln, dtsum, number, fig_dir)
    try:
        plot_set_2D(X, Y, - lapl_pot, potential, E_field, 'Poisson fields', fig_dir + 'EM_%04d' % number, no_rhs=False, axi=True)
    except:
        print('Error plot poisson')
        pass

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
    R_nodes = copy.deepcopy(Y)
    R_nodes[0] = dy / 4
    voln = compute_voln(X, dx, dy) * R_nodes

    sij = np.array([dy / 2, dx / 2])
    scale = dx * dy

    # Convection speed
    a = np.zeros((2, nny, nnx))

    # Transport coefficients
    mu, D = np.zeros_like(X), np.zeros_like(X)

    # Creation of the figures directory and numbering of outputs
    fig_dir = 'figures/' + config['casename']
    create_dir(fig_dir)
    create_dir(config['output']['folder'] + config['casename'])

    save_type = config['output']['save']
    verbose = config['output']['verbose']
    period = config['output']['period']

    file_type = config['output']['files']
    save_fig, save_data = re.search('fig', file_type), re.search('data', file_type)
    if save_data:
        data_dir = 'data/' + config['casename']
        create_dir(data_dir)

    # Timestep fixed
    dt = config['params']['dt']
    dtsum = 0

    # Background electric field
    backE = config['poisson']['backE']
    up = - x * backE
    left = np.zeros_like(y)
    right = - np.ones_like(y) * backE * xmax

    # Params
    nit = config['params']['nit']
    photo = config['params']['photoionization'] != 'no'
    if photo:
        photo_model = config['params']['photoionization']
        # Pressure in Torr
        pO2 = 150

        # Boundary conditions
        up_photo = np.zeros_like(x)
        left_photo = np.zeros_like(y)
        right_photo = np.zeros_like(y)

        mats_photo = []
        irate = np.zeros_like(X)
        Sph = np.zeros_like(X)

        if photo_model == 'two':
            for i in range(2):
                # Axisymmetric resolution
                mats_photo.append(photo_axisym(dx, dy, nnx, nny, R_nodes, (lambda_j_two[i] * pO2)**2, scale))
        elif photo_model == 'three':
            for i in range(3):
                # Axisymmetric resolution
                mats_photo.append(photo_axisym(dx, dy, nnx, nny, R_nodes, (lambda_j_three[i] * pO2)**2, scale))

    input_fn = config['input']


    if input_fn == 'none':
        number = 1

        nd = np.zeros((3, nny, nnx))
        resnd = np.zeros((3, nny, nnx))
        # Gaussian initialization for the electrons and positive ions
        n_back = config['params']['n_back']
        n_gauss = config['params']['n_gauss']
        nd[0, :] = gaussian(X, Y, n_gauss, 2e-3, 0, 2e-4, 2e-4) + n_back
        nd[1, :] = gaussian(X, Y, n_gauss, 2e-3, 0, 2e-4, 2e-4) + n_back

    else:
        # Scalar and Residual declaration
        resnd = np.zeros((3, nny, nnx))

        # Loading of densities
        nd = np.load(config['input']['nd'])

        number = int(re.search('_(\d+)\.npy', config['input']['ne']).group(1)) + 1
        dtsum = (number - 1) * config['output']['period'] * dt


    # Print header to sum up the parameters
    if verbose:
        print(f'Number of nodes: nnx = {nnx:d} -- nny = {nny:d}')
        print(f'Bounding box: ({xmin:.1e}, {ymin:.1e}), ({xmax:.1e}, {ymax:.1e})')
        print(f'dx = {dx:.2e} -- dy = {dy:.2e} -- Timestep = {dt:.2e}')
        print('------------------------------------')
        if input_fn == 'none':
            print('Start of simulation')
        else:
            print('Restart of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    # Construction of the matrix
    A = matrix_axisym(dx, dy, nnx, nny, R_nodes, scale)

    if config['output']['dl_save'] == 'yes':
        potential_list = np.zeros((nit, nny, nnx))
        physical_rhs_list = np.zeros((nit, nny, nnx))
        if photo:
            Sph_list = np.zeros((nit, nny, nnx))
            irate_list = np.zeros((nit, nny, nnx))

    # Temporal values to store (position of positive streamer, position of negative streamer, energy of the discharge)
    gstreamer = np.zeros((nit + 1, 4))
    gstreamer[:, 0] = np.linspace(0, nit * dt, nit + 1)

    # Iterations
    for it in range(1, nit + 1):
        dtsum += dt

        # Solve the Poisson equation / Axisymmetric resolution
        physical_rhs = (nd[1] - nd[0] - nd[2]).reshape(-1) * co.e / co.epsilon_0
        # physical_rhs = (nionp - ne - nn).reshape(-1) * co.e / co.epsilon_0
        rhs = - physical_rhs * scale
        dirichlet_bc_axi(rhs, nnx, nny, up, left, right)
        potential = spsolve(A, rhs).reshape(nny, nnx)
        E_field = - grad(potential, dx, dy, nnx, nny)
        physical_rhs = physical_rhs.reshape((nny, nnx))

        # Update of the residual to zero
        resnd[:] = 0

        # Application of chemistry
        if photo and it % 10 == 1:
            morrow(mu, D, E_field, nd, resnd, nnx, nny, voln, irate=irate)
            Sph[:] = 0
            print('--> Photoionization resolution')
            if photo_model == 'two':
                for i in range(2):
                    rhs = - irate.reshape(-1) * A_j_two[i] * pO2**2 * scale
                    dirichlet_bc_axi(rhs, nnx, nny, up_photo, left_photo, right_photo)
                    Sph += spsolve(mats_photo[i], rhs).reshape(nny, nnx)
            elif photo_model == 'three':
                for i in range(3):
                    rhs = - irate.reshape(-1) * A_j_three[i] * pO2**2 * scale
                    dirichlet_bc_axi(rhs, nnx, nny, up_photo, left_photo, right_photo)
                    Sph += spsolve(mats_photo[i], rhs).reshape(nny, nnx)
        else:
            morrow(mu, D, E_field, nd, resnd, nnx, nny, voln)

        if photo:
            resnd[0] -= Sph * voln
            resnd[1] -= Sph * voln

        # Convective and diffusive flux
        a = - mu * E_field
        diff_flux = D * grad(nd[0], dx, dy, nnx, nny)

        # Loop on the cells to compute the interior flux and update residuals
        compute_flux(resnd[0], a, nd[0], diff_flux, sij, ncx, ncy, r=y)
        # Boundary conditions
        outlet_y(resnd[0], a, nd[0], diff_flux, dx, -1, r=np.max(Y))
        outlet_x(resnd[0], a, nd[0], diff_flux, dy, 0, r=Y)
        outlet_x(resnd[0], a, nd[0], diff_flux, dy, -1, r=Y)


        for i in range(3):
            nd[i] = nd[i] - resnd[i] * dt / voln

        # Post processing of macro values
        normE = np.sqrt(E_field[0, :, :]**2 + E_field[1, :, :]**2)
        normE_ax = normE[0, :]
        n_middle = int(nnx / 2)
        indneg = np.argmax(normE_ax[:n_middle])
        indpos = n_middle + np.argmax(normE_ax[n_middle:])
        gstreamer[it, 1], gstreamer[it, 2] = x[np.argmax(normE_ax[:n_middle])], x[n_middle + np.argmax(normE_ax[n_middle:])]
        gstreamer[it, 3] = gstreamer[it - 1, 3] + co.e * dt * np.sum(nd[0] * mu * normE * voln)

        if verbose and (it % period == 0 or it == nit):
            print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, dt, dtsum, width=14))

        if config['output']['dl_save'] == 'yes':
            potential_list[it - 1, :, :] = potential + backE * X
            physical_rhs_list[it - 1, :, :] = physical_rhs
            if photo:
                irate_list[it - 1, :, :] = irate
                Sph_list[it - 1, :, :] = Sph

        if save_type == 'iteration':
            if it % period == 0 or it == nit:
                if save_fig:
                    plot_streamer(X, Y, nd, resnd / voln, dtsum, fig_dir + 'dens_%04d' % number)
                    plot_streamer_1D(X, Y, nd, resnd / voln, dtsum, [0, 0.25, 0.5], fig_dir + 'dens_cut_%04d' % number)
                    plot_set_2D(X, Y, physical_rhs, potential, E_field, 'Poisson fields', fig_dir + 'EM_%04d' % number, no_rhs=False, axi=True)
                    if photo:
                        plot_Sph_irate(X, Y, dx, dy, Sph, irate, nnx, nny, fig_dir + 'Sph_%04d' % number)
                if save_data:
                    np.save(data_dir + f'nd_{number:04d}', nd)

                number += 1
        elif save_type == 'none':
            pass

    plot_global(gstreamer, [xmin, xmax], fig_dir + 'globals')
    np.save(data_dir + 'globals', gstreamer)

    if config['output']['dl_save'] == 'yes':
        np.save(config['output']['folder'] + config['casename'] + 'potential.npy', potential_list)
        np.save(config['output']['folder'] + config['casename'] + 'physical_rhs.npy', physical_rhs_list)
        if photo:
            np.save(config['output']['folder'] + config['casename'] + 'Sph.npy', Sph_list)
            np.save(config['output']['folder'] + config['casename'] + 'irate.npy', irate_list)

if __name__ == '__main__':

    with open('config_streamer.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)
