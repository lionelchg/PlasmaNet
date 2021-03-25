########################################################################################################################
#                                                                                                                      #
#                                       Filter a rhs_random dataset using FFTs                                         #
#                                                                                                                      #
#                                      Guillaume Bogopolsky, CERFACS, 26.05.2020                                       #
#                                                                                                                      #
########################################################################################################################

import os
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import matplotlib
from numba import njit

from PlasmaNet.common.plot import plot_set_2D
from PlasmaNet.common.operators_numpy import grad
from PlasmaNet.poissonsolver.linsystem import cartesian_matrix, impose_dirichlet
from scipy.sparse.linalg import spsolve

matplotlib.use('Agg')

# Hardcoded parameters
domain_length = 0.01
plot_period = 1000


def dataset_filter(dataset, input_cutoff_frequency, mesh_size, domain_length):
    """
    Filters in-place a dataset by cutting of frequencies in a Fourier transform.

    Parameters
    ----------
    dataset : Numpy array
            Dataset in Numpy format, of shape [dataset_size, Ny, Nz]

    input_cutoff_frequency : float
            Cutoff frequency in both directions

    mesh_size : int
            Size of the square mesh

    domain_length : float
            Physical length of the domain
    """
    for i in tqdm(range(dataset.shape[0])):
        transf = np.fft.fft2(dataset[i])
        freq = np.fft.fftfreq(mesh_size, domain_length / mesh_size)
        cutoff_fourier_space(transf, freq, input_cutoff_frequency)
        dataset[i] = np.real(np.fft.ifft2(transf))


def dataset_clean_filter(dataset, input_cutoff_frequency, mesh_size, domain_length):
    """
    Similar to dataset_filter, but realises a clean frequency cut by mirroring the input
    on the four quadrants before the FFT.
    """
    for i in tqdm(range(dataset.shape[0])):
        # Mirror the input
        mirror = np.zeros((2 * mesh_size - 1, 2 * mesh_size - 1))
        mirror[mesh_size - 1 :, mesh_size - 1 :] = dataset[i]
        mirror[: mesh_size - 1, : mesh_size - 1] = dataset[i, -1:0:-1, -1:0:-1]
        mirror[mesh_size - 1 :, : mesh_size - 1] = - dataset[i, :, -1:0:-1]
        mirror[: mesh_size - 1, mesh_size - 1 :] = - dataset[i, -1:0:-1, :]

        # Do the Fourier transform and cutoff
        transf = np.fft.fft2(mirror)
        freq = np.fft.fftfreq(mesh_size * 2 - 1, domain_length / mesh_size)
        cutoff_fourier_space(transf, freq, input_cutoff_frequency)
        # Replace the input with the correct quadrant
        dataset[i] = np.real(np.fft.ifft2(transf))[mesh_size - 1 :, mesh_size - 1 :]


@njit
def cutoff_fourier_space(transf, freq, cutoff_frequency):
    """ Iterates over Fourier space and sets to zero all frequencies above cutoff_frequency. """
    size = transf.shape[0]
    for k in range(size):
        for j in range(size):
            if np.abs(freq[k]) > cutoff_frequency or np.abs(freq[j]) > cutoff_frequency:
                transf[k, j] = 0


def poisson_solver(physical_rhs, dx, n_points):
    """
    Solves the Poisson equation for the given physical_rhs with Dirichlet boundary conditions.
    """

    tmp_rhs = - physical_rhs.reshape(-1) * dx**2
    # Imposing Dirichlet boundary conditions
    zeros_bc = np.zeros(n_points)
    dirichlet_bc(rhs, n_points, zeros_bc, zeros_bc, zeros_bc, zeros_bc)
    # Solving the sparse linear system
    tmp_potential = spsolve(A, tmp_rhs).reshape(n_points, n_points)

    return tmp_potential


if __name__ == '__main__':
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Filter a rhs_random dataset into a new dataset")
    parser.add_argument('dataset', type=Path, help='Input dataset path')
    parser.add_argument('cut_off_frequency', type=float, help='Cutoff frequency [Hz]')
    parser.add_argument('--split', type=float, default=1.0, help='Filter a fraction of the input dataset (default: 1)')
    parser.add_argument('--plot', action='store_true', help='Activate plots')
    parser.add_argument('--filter_target', action='store_true', help='Filter only input by default')

    args = parser.parse_args()

    # Load the input dataset
    potential = np.load(args.dataset / 'potential.npy')
    rhs = np.load(args.dataset / 'physical_rhs.npy')

    mesh_size = rhs.shape[1]
    print(f'Mesh size : {mesh_size}')
    print(f'Dataset size : {rhs.shape[0]}')

    # If split is different from one, work only on the first work_length elements
    if args.split != 1.0:
        work_length = int(rhs.shape[0] * args.split)
        print(f'Filtering the first {work_length} patches')
        dataset_clean_filter(rhs[:work_length], args.cut_off_frequency, mesh_size, domain_length)
        if args.filter_target:
            dataset_clean_filter(potential[:work_length], args.cut_off_frequency, mesh_size, domain_length)
    else:
        dataset_clean_filter(rhs, args.cut_off_frequency, mesh_size, domain_length)
        if args.filter_target:
            dataset_clean_filter(potential, args.cut_off_frequency, mesh_size, domain_length)

    # Determine new dataset name
    new_name = args.dataset.name
    if args.split != 1.0:
        new_name += f'_split{args.split:.1f}'
    if args.filter_target:
        new_name += f'_both'
    new_name += f'_filtered{args.cut_off_frequency:.0f}Hz'

    new_path = args.dataset.with_name(new_name)  # Return new Path object with changed name
    new_path.mkdir()
    print(new_path.as_posix())

    bcs = 'dirichlet'

    # Plot if asked
    if args.plot:
        dx = domain_length / (mesh_size - 1)
        x = np.linspace(0, domain_length, mesh_size)
        X, Y = np.meshgrid(x, x)
        A = cartesian_matrix(dx, dx, mesh_size, mesh_size, dx * dx, bcs)
        fig_dir = 'figures/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        for i in tqdm(range(rhs.shape[0])):
            if i % plot_period == 0:
                E_field = - grad(potential[i], dx, dx, mesh_size, mesh_size)
                plot_set_2D(X, Y, rhs[i], potential[i], E_field, f'Filtered {i}', fig_dir + f'filtered_{i:05d}')
                # comp_pot = poisson_solver(rhs[i], dx, mesh_size)
                # E_field = - grad(comp_pot, dx, dx, mesh_size, mesh_size)
                # plot_set_2D(X, Y, rhs[i], comp_pot, E_field, f'Filtered {i}', fig_dir + f'filtered_computed_{i:05d}')

    # Save the new dataset
    np.save(new_path / 'potential.npy', potential)
    np.save(new_path / 'physical_rhs.npy', rhs)
