########################################################################################################################
#                                                                                                                      #
#                                       Interpolate a random_rhs dataset                                               #
#                                                                                                                      #
#                                       Ekhi AJuria, CERFACS, 26.05.2020                                               #
#                                                                                                                      #
########################################################################################################################

import os
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import get_context

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from numba import njit

from poissonsolver.plot import plot_set_2D
from poissonsolver.operators import grad
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from scipy.sparse.linalg import spsolve
from scipy import interpolate

matplotlib.use('Agg')

# Hardcoded parameters
domain_length = 0.01
plot_period = 1000

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier

def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu', max_value=None):
    if colormap == 'RdBu' and max_value is None:
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
    elif colormap == 'RdBu':
        levels = np.linspace(- max_value, max_value, 101)
    else:
        levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap=colormap)
    fig.colorbar(cs1, ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)

def plot_fields(POT, q_s, q, h_s, h, index_b, fig_path):

    # Lots of plots
    fig, axes = plt.subplots(figsize=(25, 4), nrows=1, ncols=5)
    fig.suptitle('Example field {}'.format(index_b), fontsize=16, y=0.95)

    # Same scale for output and target
    plot_ax_scalar(fig, axes[0], np.arange(len(POT[0,:])), np.arange(len(POT[:,0])), POT, 'Original potential')
    plot_ax_scalar(fig, axes[1], np.arange(len(q_s[0,:])), np.arange(len(q_s[:,0])), q_s, 'Quarter Small')
    plot_ax_scalar(fig, axes[2], np.arange(len(q[0,:])), np.arange(len(q[:,0])), q, 'Quarter Big Difference')
    plot_ax_scalar(fig, axes[3], np.arange(len(h_s[0,:])), np.arange(len(h_s[:,0])), h_s, 'Half Small Difference')
    plot_ax_scalar(fig, axes[4], np.arange(len(h[0,:])), np.arange(len(h[:,0])), h, 'Hald Big')

    plt.savefig(fig_path / 'fig_example_{}.png'.format(index_b),dpi=300)
    plt.close()

def dataset_clean_filter(dataset, mesh_size, domain_length):
    """
    Similar to dataset_filter, but realises a clean frequency cut by mirroring the input
    on the four quadrants before the FFT.
    """
    # Mirror the input
    mirror = np.zeros((2 * mesh_size - 1, 2 * mesh_size - 1))
    mirror[mesh_size - 1 :, mesh_size - 1 :] = dataset
    mirror[: mesh_size - 1, : mesh_size - 1] = dataset[ -1:0:-1, -1:0:-1]
    mirror[mesh_size - 1 :, : mesh_size - 1] = - dataset[:, -1:0:-1]
    mirror[: mesh_size - 1, mesh_size - 1 :] = - dataset[-1:0:-1, :]

    # Do the Fourier transform and cutoff
    transf = np.fft.fft2(mirror)
    freq = np.fft.fftfreq(mesh_size * 2 - 1, domain_length / mesh_size)
    Freq = np.fft.fftshift(np.fft.fftfreq(mesh_size * 2 - 1, domain_length / mesh_size))


    return np.abs(np.fft.fftshift(transf)), Freq

def plot_a_f_fields(n, n_f, n2, n2_f, n4, n4_f, index_b, log_t, diff):
    axes_max = 750
    ones = np.ones_like(n)
    # Lots of plots
    fig, ax = plt.subplots(figsize=(25, 4), nrows=1, ncols=3)
    #fig.suptitle('Example field FFT {}'.format(index_b),  y=0.95)

    if log_t:
        #cs1 = ax[0].contourf(n_f, n_f, np.maximum(-2*ones,np.log(n)), 100, cmap='Blues')
        cs1 = ax[0].contourf(n_f, n_f, np.maximum(-2*ones,np.log(np.maximum(0.0001*ones,n))), 100, cmap='Blues')
    else: 
        cs1 = ax[0].contourf(n_f, n_f, n, 100, cmap='Blues')
    if diff:
        ax[0].set_title('Original scale')
    else:
        ax[0].set_title('Original scale')
    #ax[0].set_xlim([-axes_max, axes_max])
    #ax[0].set_ylim([-axes_max, axes_max])
    ax[0].set_aspect('equal')
    fig.colorbar(cs1, ax=ax[0])

    if log_t:
        cs2 = ax[1].contourf(n2_f, n2_f, np.maximum(-2*ones,np.log(np.maximum(0.0001*ones,n2))), 100, cmap='Blues')
        #cs2 = ax[1].contourf(n2_f, n2_f, np.maximum(-2*ones,np.log(n2)), 100, cmap='Blues')
    else:
        cs2 = ax[1].contourf(n2_f, n2_f, n2, 100, cmap='Blues') 

    if diff:
        ax[1].set_title('Half scale Difference')
    else:
        ax[1].set_title('Half scale')
    #ax[1].set_xlim([-axes_max, axes_max])
    #ax[1].set_ylim([-axes_max, axes_max])
    ax[1].set_aspect('equal')
    fig.colorbar(cs2, ax=ax[1])

    if log_t:
        cs3 = ax[2].contourf(n4_f, n4_f, np.maximum(-2*ones,np.log(np.maximum(0.0001*ones,n4))), 100, cmap='Blues')
    else:
        cs3 = ax[2].contourf(n4_f, n4_f, n4, 100, cmap='Blues')

    if diff:
        ax[2].set_title('Quarter scale Difference')
    else:
        ax[2].set_title('Quarter scale')
    #ax[2].set_xlim([-axes_max, axes_max])
    #ax[2].set_ylim([-axes_max, axes_max])
    ax[2].set_aspect('equal')
    fig.colorbar(cs3, ax=ax[2])

    if log_t:
        if diff:
            plt.savefig(fig_path / 'fig_Diff_FFT_log_{}.png'.format(index_b),dpi=300)
        else:
            plt.savefig(fig_path / 'fig_FFT_log_{}.png'.format(index_b),dpi=300)
    else:
        if diff:
            plt.savefig(fig_path / 'fig_Diff_FFT_{}.png'.format(index_b),dpi=300)
        else:
            plt.savefig(fig_path / 'fig_FFT_{}.png'.format(index_b),dpi=300)
    plt.close()

def big_loop(potential, pot_quarter_orig, pot_half_orig, fig_path):
    for i in tqdm(range(len(potential[:,0,0]))):
        # Interpolate
        xx_o = np.arange(len(potential[i,:,0]))/(len(potential[i,:,0])-1)
        yy_o = np.arange(len(potential[i,0,:]))/(len(potential[i,:,0])-1)

        f = interpolate.interp2d(xx_o, yy_o, potential[i], kind='cubic')

        pot_q = potential[i,::4,::4]
        pot_h = potential[i,::2,::2]

        xx_q = np.arange(len(pot_q[:,0]))/(len(pot_q[:,0])-1)
        yy_q = np.arange(len(pot_q[0,:]))/(len(pot_q[0,:])-1)

        xx_h = np.arange(len(pot_h[:,0]))/(len(pot_h[:,0])-1)
        yy_h = np.arange(len(pot_h[0,:]))/(len(pot_h[0,:])-1)

        pot_quarter = f(xx_q, yy_q)
        pot_half  = f(xx_h, yy_h)

        f_h = interpolate.interp2d(xx_h, yy_h, pot_half, kind='linear')
        f_q = interpolate.interp2d(xx_q, yy_q, pot_quarter, kind='linear')

        pot_quarter_orig[i] = f_q(xx_o, yy_o)
        pot_half_orig[i] = f_h(xx_o, yy_o)

        if i % 100  == 0:
            #print('Percentage Done : {:.0f} %'.format(100*i/len(potential[:,0,0])))
            plot_fields(potential[i], pot_quarter, pot_quarter_orig[i]-potential[i], pot_half, pot_half_orig[i]-potential[i], i, fig_path)
            n_A, n_f = dataset_clean_filter(potential[i], mesh_size, domain_length)
            n2_A, n2_f = dataset_clean_filter(pot_half_orig[i], mesh_size, domain_length)
            n4_A, n4_f = dataset_clean_filter(pot_quarter_orig[i], mesh_size, domain_length)

            n2_diff_A, n2_diff_f = dataset_clean_filter(potential[i] - pot_half_orig[i], mesh_size, domain_length)
            n4_diff_A, n4_diff_f = dataset_clean_filter(potential[i] - pot_quarter_orig[i], mesh_size, domain_length)

            plot_a_f_fields(n_A, n_f, n2_A, n2_f, n4_A, n4_f, i, True, False)
            plot_a_f_fields(n_A, n_f, n2_A, n2_f, n4_A, n4_f, i, False, False)

            plot_a_f_fields(n_A, n_f, n2_diff_A, n2_diff_f, n4_diff_A, n4_diff_f, i, True, True)
            plot_a_f_fields(n_A, n_f, n2_diff_A, n2_diff_f, n4_diff_A, n4_diff_f, i, False, True)
    return pot_half_orig, pot_quarter_orig

if __name__ == '__main__':
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Filter a rhs_random dataset into a new dataset")
    parser.add_argument('dataset', type=Path, help='Input dataset path')

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
    new_name += '_interpolated_3_scales'
    fig_fol = args.dataset.name + 'figures'

    new_path = args.dataset.with_name(new_name)  # Return new Path object with changed name
    fig_path = args.dataset.with_name(fig_fol)  # Return new Path object with changed name
    if not os.path.exists(new_path):
        new_path.mkdir()
    if not os.path.exists(fig_path):
        fig_path.mkdir()
    print(new_path.as_posix())
    print(fig_path.as_posix())

    pot_quarter_orig = np.zeros_like(potential)
    pot_half_orig =  np.zeros_like(potential)


    pot_half_orig, pot_quarter_orig = big_loop(potential, pot_quarter_orig, pot_half_orig, fig_path)

    potential_expand = np.expand_dims(potential, axis=1)
    pot_half_orig = np.expand_dims(pot_half_orig, axis=1)
    pot_quarter_orig = np.expand_dims(pot_quarter_orig, axis=1)
    new_pot = np.concatenate((potential_expand, pot_half_orig, pot_quarter_orig), axis = 1)

    # Save the new dataset
    np.save(new_path / 'potential_interp.npy', new_pot)
    np.save(new_path / 'potential.npy', potential)
    np.save(new_path / 'physical_rhs.npy', rhs)
