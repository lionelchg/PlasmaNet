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

def plot_fields_all(POT, p_2, p_4, p_8, p_16, p_32, index_b, fig_path, not_orig):

    # Lots of plots
    fig, axes = plt.subplots(figsize=(30, 4), nrows=1, ncols=6)
    fig.suptitle('Example field {}'.format(index_b), fontsize=16, y=0.95)

    # Same scale for output and target
    plot_ax_scalar(fig, axes[0], np.arange(len(POT[0,:])), np.arange(len(POT[:,0])), POT, 'Original potential')
    plot_ax_scalar(fig, axes[1], np.arange(len(p_2[0,:])), np.arange(len(p_2[:,0])), p_2, 'N/2 Scale')
    plot_ax_scalar(fig, axes[2], np.arange(len(p_4[0,:])), np.arange(len(p_4[:,0])), p_4, 'N/4 Scale')
    plot_ax_scalar(fig, axes[3], np.arange(len(p_8[0,:])), np.arange(len(p_8[:,0])), p_8, 'N/8 Scale')
    plot_ax_scalar(fig, axes[4], np.arange(len(p_16[0,:])), np.arange(len(p_16[:,0])), p_16, 'N/16 Scale')
    plot_ax_scalar(fig, axes[5], np.arange(len(p_32[0,:])), np.arange(len(p_32[:,0])), p_32, 'N/32 Scale')

    if not_orig:
        plt.savefig(fig_path / 'fig_all_example_{}.png'.format(index_b),dpi=300)
    else:
        plt.savefig(fig_path / 'fig_all_orig_example_{}.png'.format(index_b),dpi=300)
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

def inside_plot(fig,axis,n,n_f, ones, log_t, diff, title, limit, axes_max):
    if log_t:
        #cs1 = ax[0].contourf(n_f, n_f, np.maximum(-2*ones,np.log(n)), 100, cmap='Blues')
        cs1 = axis.contourf(n_f, n_f, np.maximum(-0.5*ones,np.log(np.maximum(0.0001*ones,n))), 100, cmap='Blues')
    else:
        cs1 = axis.contourf(n_f, n_f, n, 100, cmap='Blues')
    if diff:
        axis.set_title(title + ' Diff')
    else:
        axis.set_title(title)
    if limit:
        axis.set_xlim([-axes_max, axes_max])
        axis.set_ylim([-axes_max, axes_max])
    axis.set_aspect('equal')
    fig.colorbar(cs1, ax=axis)


def plot_a_f_fields(n, n_f, n2, n2_f, n4, n4_f,  n8, n8_f, n16, n16_f, n32, n32_f, index_b, log_t, diff):
    axes_max = 750
    limit = True
    ones = np.ones_like(n)
    # Lots of plots
    fig, ax = plt.subplots(figsize=(50, 10), nrows=1, ncols=6)
    #fig.suptitle('Example field FFT {}'.format(index_b),  y=0.95)

    inside_plot( fig, ax[0], n, n_f, ones, log_t, diff, 'Original Scale', limit, axes_max)
    inside_plot( fig, ax[1], n2, n2_f, ones, log_t, diff, 'N/2 Scale', limit, axes_max)
    inside_plot( fig, ax[2], n4, n4_f, ones, log_t, diff, 'N/4 Scale', limit, axes_max)
    inside_plot( fig, ax[3], n8, n8_f, ones, log_t, diff, 'N/8 Scale', limit, axes_max)
    inside_plot( fig, ax[4], n16, n16_f, ones, log_t, diff, 'N/16 Scale', limit, axes_max)
    inside_plot( fig, ax[5], n32, n32_f, ones, log_t, diff, 'N/32 Scale', limit, axes_max)

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



def big_loop(potential, pot_quarter_orig, pot_half_orig, pot_8_orig, pot_16_orig, pot_32_orig, fig_path):
    for i in tqdm(range(len(potential[:,0,0]))):
        # Interpolate
        xx_o = np.arange(len(potential[i,:,0]))/(len(potential[i,:,0])-1)
        yy_o = np.arange(len(potential[i,0,:]))/(len(potential[i,:,0])-1)

        f = interpolate.interp2d(xx_o, yy_o, potential[i], kind='cubic')

        pot_32 = potential[i,::32,::32]
        pot_16 = potential[i,::16,::16]
        pot_8 = potential[i,::8,::8]
        pot_q = potential[i,::4,::4]
        pot_h = potential[i,::2,::2]

        xx_q = np.arange(len(pot_q[:,0]))/(len(pot_q[:,0])-1)
        yy_q = np.arange(len(pot_q[0,:]))/(len(pot_q[0,:])-1)

        xx_h = np.arange(len(pot_h[:,0]))/(len(pot_h[:,0])-1)
        yy_h = np.arange(len(pot_h[0,:]))/(len(pot_h[0,:])-1)

        xx_8 = np.arange(len(pot_8[:,0]))/(len(pot_8[:,0])-1)
        yy_8 = np.arange(len(pot_8[0,:]))/(len(pot_8[0,:])-1)

        xx_16 = np.arange(len(pot_16[:,0]))/(len(pot_16[:,0])-1)
        yy_16 = np.arange(len(pot_16[0,:]))/(len(pot_16[0,:])-1)

        xx_32 = np.arange(len(pot_32[:,0]))/(len(pot_32[:,0])-1)
        yy_32 = np.arange(len(pot_32[0,:]))/(len(pot_32[0,:])-1)

        pot_quarter = f(xx_q, yy_q)
        pot_half  = f(xx_h, yy_h)
        pot_8 = f(xx_8, yy_8)
        pot_16 = f(xx_16, yy_16)
        pot_32 = f(xx_32, yy_32)

        interpo = 'cubic'
        f_h = interpolate.interp2d(xx_h, yy_h, pot_half, kind=interpo)
        f_q = interpolate.interp2d(xx_q, yy_q, pot_quarter, kind=interpo)
        f_8 = interpolate.interp2d(xx_8, yy_8, pot_8, kind=interpo)
        f_16 = interpolate.interp2d(xx_16, yy_16, pot_16, kind=interpo)
        f_32 = interpolate.interp2d(xx_32, yy_32, pot_32, kind=interpo)

        pot_quarter_orig[i] = f_q(xx_o, yy_o)
        pot_half_orig[i] = f_h(xx_o, yy_o)
        pot_8_orig[i] = f_8(xx_o, yy_o)
        pot_16_orig[i] = f_16(xx_o, yy_o)
        pot_32_orig[i] = f_32(xx_o, yy_o)

        if i % 100  == 0:
            #print('Percentage Done : {:.0f} %'.format(100*i/len(potential[:,0,0])))
            plot_fields_all(potential[i], pot_half, pot_quarter, pot_8, pot_16, pot_32, i, fig_path, True)
            plot_fields_all(potential[i], pot_half_orig[i], pot_quarter_orig[i], pot_8_orig[i], pot_16_orig[i], pot_32_orig[i], i, fig_path, False)
            plot_fields(potential[i], pot_quarter, pot_quarter_orig[i]-potential[i], pot_half, pot_half_orig[i]-potential[i], i, fig_path)
            n_A, n_f = dataset_clean_filter(potential[i], mesh_size, domain_length)
            n2_A, n2_f = dataset_clean_filter(pot_half_orig[i], mesh_size, domain_length)
            n4_A, n4_f = dataset_clean_filter(pot_quarter_orig[i], mesh_size, domain_length)
            n8_A, n8_f = dataset_clean_filter(pot_8_orig[i], mesh_size, domain_length)
            n16_A, n16_f = dataset_clean_filter(pot_16_orig[i], mesh_size, domain_length)
            n32_A, n32_f = dataset_clean_filter(pot_32_orig[i], mesh_size, domain_length)

            n2_diff_A, n2_diff_f = dataset_clean_filter(potential[i] - pot_half_orig[i], mesh_size, domain_length)
            n4_diff_A, n4_diff_f = dataset_clean_filter(potential[i] - pot_quarter_orig[i], mesh_size, domain_length)
            n8_diff_A, n8_diff_f = dataset_clean_filter(potential[i] - pot_8_orig[i], mesh_size, domain_length)
            n16_diff_A, n16_diff_f = dataset_clean_filter(potential[i] - pot_16_orig[i], mesh_size, domain_length)
            n32_diff_A, n32_diff_f = dataset_clean_filter(potential[i] - pot_32_orig[i], mesh_size, domain_length)

            plot_a_f_fields(n_A, n_f, n2_A, n2_f, n4_A, n4_f,  n8_A, n8_f, n16_A, n16_f, n32_A, n32_f, i, True, False)
            plot_a_f_fields(n_A, n_f, n2_A, n2_f, n4_A, n4_f,  n8_A, n8_f, n16_A, n16_f, n32_A, n32_f, i, False, False)
            plot_a_f_fields(n_A, n_f, n2_diff_A, n2_diff_f, n4_diff_A, n4_diff_f,  n8_diff_A, n8_diff_f, n16_diff_A, n16_diff_f, n32_diff_A, n32_diff_f, i, True, True)
            plot_a_f_fields(n_A, n_f, n2_diff_A, n2_diff_f, n4_diff_A, n4_diff_f,  n8_diff_A, n8_diff_f, n16_diff_A, n16_diff_f, n32_diff_A, n32_diff_f, i, False, True)

    return pot_half_orig, pot_quarter_orig

if __name__ == '__main__':
    # CLI argument parser
    parser = argparse.ArgumentParser(description="Filter a rhs_random dataset into a new dataset")
    parser.add_argument('dataset', type=Path, help='Input dataset path')

    args = parser.parse_args()

    # Load the input dataset
    potential = np.load(args.dataset / 'potential.npy')[:200]
    rhs = np.load(args.dataset / 'physical_rhs.npy')[:200]

    mesh_size = rhs.shape[1]

    print(f'Mesh size : {mesh_size}')
    print(f'Dataset size : {rhs.shape[0]}')
    print(f'Shape of potential field : {potential.shape}')


    # Determine new dataset name
    new_name = args.dataset.name
    new_name += '_Small_interpolated_3_scales'
    fig_fol = args.dataset.name + '_Small_figures'

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
    pot_8_orig =  np.zeros_like(potential)
    pot_16_orig =  np.zeros_like(potential)
    pot_32_orig =  np.zeros_like(potential)

    pot_half_orig, pot_quarter_orig = big_loop(potential, pot_quarter_orig, pot_half_orig, pot_8_orig, pot_16_orig, pot_32_orig, fig_path)

    potential_expand = np.expand_dims(potential, axis=1)
    pot_half_orig = np.expand_dims(pot_half_orig, axis=1)
    pot_quarter_orig = np.expand_dims(pot_quarter_orig, axis=1)
    new_pot = np.concatenate((potential_expand, pot_half_orig, pot_quarter_orig), axis = 1)

    # Save the new dataset
    np.save(new_path / 'potential_interp.npy', new_pot)
    np.save(new_path / 'potential.npy', potential)
    np.save(new_path / 'physical_rhs.npy', rhs)
