########################################################################################################################
#                                                                                                                      #
#                                      Separation of the Laplace and rhs problem                                       #
#                                                 for Streamer dataset                                                 #
#                                                                                                                      #
#                                 Ekhi Ajuria, Guillaume Bogopolsky CERFACS, 30.09.2020                                #
#                                                                                                                      #
########################################################################################################################

import argparse
import copy
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from poissonsolver.plot import plot_set_1D, plot_set_2D, plot_potential
from poissonsolver.linsystem import matrix_cart, matrix_axisym, laplace_square_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

matplotlib.use('Agg')


def compute_voln(X, dx, dy):
    """ Computes the nodal volume associated to each node (i, j). """
    voln = np.ones_like(X) * dx * dy
    voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
        dx * dy / 2, dx * dy / 2, dx * dy / 2, dx * dy / 2
    voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
        dx * dy / 4, dx * dy / 4, dx * dy / 4, dx * dy / 4
    return voln


def inside_plot(fig, axis, n_x, n_y, n, ones, log_t, title, limit, axes_max_x, axes_max_y):
    if log_t:
        # cs1 = ax[0].contourf(n_f, n_f, np.maximum(-2*ones,np.log(n)), 100, cmap='Blues')
        cs1 = axis.contourf(n_x, n_y, np.maximum(-0.5*ones, np.log(np.maximum(0.0001*ones, n))), 100, cmap='Blues')
    else:
        cs1 = axis.contourf(n_x, n_y, n, 100, cmap='Blues')

    axis.set_title(title)

    if limit:
        axis.set_xlim([0, axes_max_x])
        axis.set_ylim([0, axes_max_y])
    axis.set_aspect('equal')
    fig.colorbar(cs1, ax=axis)


def plot_fields(n_x, n_y, rhs, potential, potential_rhs, potential_bc, potential_solved, index_b, log_t, initial):
    axes_max_x = np.max(n_x)
    axes_max_y = np.max(n_y) 
    limit = True
    ones = np.ones_like(n_x)

    # Depending on initial or not, 2 or 5 images will be plotted
    if initial:
        size_big = 10
        ncols = 2
    else:
        size_big = 25
        ncols = 5

    fig, ax = plt.subplots(figsize=(size_big, 5), nrows=1, ncols=ncols)

    inside_plot(fig, ax[0], n_x, n_y, rhs[index_b], ones, log_t,  'rhs', limit, axes_max_x, axes_max_y)
    inside_plot(fig, ax[1], n_x, n_y, potential[index_b], ones, log_t,  'Initial Potential',
                limit, axes_max_x, axes_max_y)

    if not initial:
        inside_plot(fig, ax[2], n_x, n_y, potential_rhs[index_b], ones, log_t,  'Potential rhs',
                    limit, axes_max_x, axes_max_y)
        inside_plot(fig, ax[3], n_x, n_y, potential_bc[index_b], ones, log_t, 'Potential BC',
                    limit, axes_max_x, axes_max_y)
        inside_plot(fig, ax[4], n_x, n_y, potential_solved[index_b], ones, log_t,  'Potential solved',
                    limit, axes_max_x, axes_max_y)

    if initial:
        plt.savefig(fig_path / 'Potential_initial_fields_{}.png'.format(index_b),dpi=300)
    else:
        plt.savefig(fig_path / 'Potential_corrected_fields_{}.png'.format(index_b),dpi=300)
    plt.close()


def separate_potential(potential, physical_rhs, A):
    """ Separate the problem in the Laplace and Poisson problems. Basically does the big loop. """
    potential_bc = np.zeros_like(potential)
    potential_rhs = np.zeros_like(potential) 
    potential_solved = np.zeros_like(potential)  

    for i in tqdm(range(potential.shape[0])):
        # Get BC
        left_bc = potential[i, :, 0]
        right_bc = potential[i, :, -1]
        sup_bc = potential[i, -1, :]
        inf_bc = potential[i, 0, :]

        # Just in case
        max_left = np.rint(np.mean(left_bc))
        max_right = np.rint(np.mean(right_bc))
        lateral_bc = np.linspace(max_left, max_right, potential.shape[2])

        # Expand the Lateral BC as the resulting linear potential field
        potential_bc[i] = np.repeat(sup_bc[np.newaxis, :], potential.shape[1], axis=0)

        # Hard copy the left and right (just in case corners are weirdly solved)
        potential_bc[i, :, 0] = left_bc
        potential_bc[i, :, -1] = right_bc
        # potential_bc[i, -1, :] = sup_bc
        # potential_bc[i, 0, :]  = inf_bc

        # Create the new rhs potential
        potential_rhs[i] = potential[i] - potential_bc[i]

        # Check the new rhs corresponds to 
        if args.use_mkl:
            potential_solved[i] = sparse_qr_solve_mkl(A, physical_rhs[i].reshape(-1) * - scale).reshape(nny, nnx)
        else:
            potential_solved[i] = spsolve(A, physical_rhs[i].reshape(-1) * - scale).reshape(nny, nnx)

        # Testing
        # np.testing.assert_allclose(potential_rhs[i], potential_solved[i], atol=1e-07, equal_nan=True, verbose=True)

    return potential_rhs, potential_bc, potential_solved


if __name__ == '__main__':
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Separate the rhs datasets into a Laplace (Dirichlet BC) "
                                                 "and Poisson problem")
    parser.add_argument("dataset", type=Path, help="Input dataset path")
    parser.add_argument("lx", type=float, help="Domain length along x")
    parser.add_argument("ly", type=float, help="Domain length along y")
    parser.add_argument("--axisym", action="store_true", help="Axisymmetric dataset")
    parser.add_argument("--plot", action="store_true", help="Execute some plots")
    parser.add_argument("--plot_period", type=int, default=50, help="Plot period (default: 50)")
    parser.add_argument("--use_mkl", action="store_true",
                        help="Use MKL sparse solver as suitesparse is not available on kraken")
    args = parser.parse_args()

    # Load the input dataset
    potential = np.load(args.dataset / 'potential.npy')
    rhs = np.load(args.dataset / 'physical_rhs.npy')

    mesh_size = rhs.shape[1]

    print(f'Mesh size : {mesh_size}')
    print(f'Dataset size : {rhs.shape[0]}')
    print(f'Shape of potential field : {potential.shape}')

    # Determine new dataset name
    new_name = args.dataset.name
    new_name += '_separated_BC_rhs'
    fig_fol = args.dataset.name + '_BC_rhs_figures'

    new_path = args.dataset.with_name(new_name)  # Return new Path object with changed name
    fig_path = args.dataset.with_name(fig_fol)  # Return new Path object with changed name

    if not os.path.exists(new_path):
        new_path.mkdir()
    if not os.path.exists(fig_path) and args.plot:
        fig_path.mkdir()
    print(f"new_path : {new_path}")
    print(f"fig_path : {fig_path}")

    # nx and ny definition
    n_x = np.arange(0, potential.shape[2])
    n_y = np.arange(0, potential.shape[1])

    potential_rhs_0 = np.zeros_like(potential)
    potential_BC_0 = np.zeros_like(potential) 
    potential_solved_0 = np.zeros_like(potential)
    index_b = 1
    log_t = False
    initial = True
    verbose = True

    if args.plot:
        # Index definition for first plot
        plot_fields(n_x, n_y, rhs, potential, potential_rhs_0, potential_BC_0, potential_solved_0,
                    index_b, log_t, initial)

    # Matrix A creation
    nnx, nny = potential.shape[2], potential.shape[1]
    ncx, ncy = nnx - 1, nny - 1  # Number of cells
    xmin, xmax = 0, args.lx
    ymin, ymax = 0, args.ly
    Lx, Ly = xmax - xmin, ymax - ymin
    dx = (xmax - xmin) / ncx
    dy = (ymax - ymin) / ncy
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)

    # Grid construction
    X, Y = np.meshgrid(x, y)
    voln = compute_voln(X, dx, dy)
    if args.axisym:
        R_nodes = copy.deepcopy(Y)
        R_nodes[0] = dy / 4
        voln *= R_nodes
    scale = dx * dy

    # Print header to sum up the parameters
    if verbose:
        print(f'Number of nodes: nnx = {nnx:d} -- nny = {nny:d}')
        print(f'Bounding box: ({xmin:.1e}, {ymin:.1e}), ({xmax:.1e}, {ymax:.1e})')
        print(f'dx = {dx:.2e} -- dy = {dy:.2e} ')
        print('------------------------------------')

    # Construction of the matrix
    if args.axisym:
        A = matrix_axisym(dx, dy, nnx, nny, R_nodes, scale)
    else:
        # A = matrix_cart(dx, dy, nnx, nny, scale)
        A = laplace_square_matrix(nnx)
    if args.use_mkl:
        from scipy.sparse import csr_matrix
        from sparse_dot_mkl import sparse_qr_solve_mkl
        A = csr_matrix(A)

    # Main loop
    pot_rhs, pot_bc, pot_solved = separate_potential(potential, rhs, A)

    if args.plot:
        # Plot some fields after the correction
        initial = False
        for index_k in range(5):
            plot_fields(n_x, n_y, rhs, potential, pot_rhs, pot_bc, pot_solved, index_k, log_t, initial)

    # Save the new dataset
    np.save(new_path / 'potential_orig.npy', potential)
    np.save(new_path / 'potential_rhs.npy', pot_rhs)
    np.save(new_path / 'potential_BC.npy', pot_bc)
    np.save(new_path / 'rhs.npy', rhs)
