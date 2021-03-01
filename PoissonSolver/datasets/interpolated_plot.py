########################################################################################################################
#                                                                                                                      #
#                             Plot functions for interpolating a random_rhs dataset                                    #
#                                                                                                                      #
#                                       Ekhi AJuria, CERFACS, 01.03.2021                                               #
#                                                                                                                      #
########################################################################################################################

'''
Plot functions used in rhs_interpolated_small.py
'''

import os
import argparse
from pathlib import Path
from tqdm import tqdm

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

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return np.ceil(n * multiplier) / multiplier
 
def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu', max_value=None):
    """
    Plotting function, defines the colormap and performs a contourf
    """
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
    """
    Useful plot functions that plot the fields: 1/1, 1/2, 1/4, 
    as well as the difference between the interpolated and the original fields
    """

    # Lots of plots
    fig, axes = plt.subplots(figsize=(25, 4), nrows=1, ncols=5)
    fig.suptitle('Example field {}'.format(index_b), fontsize=16, y=0.95)

    # Same scale for output and target
    plot_ax_scalar(fig, axes[0], np.arange(len(POT[0,:])), np.arange(len(POT[:,0])), POT, 'Original potential')
    plot_ax_scalar(fig, axes[1], np.arange(len(q_s[0,:])), np.arange(len(q_s[:,0])), q_s, 'Quarter Small')
    plot_ax_scalar(fig, axes[2], np.arange(len(q[0,:])), np.arange(len(q[:,0])), q, 'Quarter Big Difference')
    plot_ax_scalar(fig, axes[3], np.arange(len(h_s[0,:])), np.arange(len(h_s[:,0])), h_s, 'Half Small Difference')
    plot_ax_scalar(fig, axes[4], np.arange(len(h[0,:])), np.arange(len(h[:,0])), h, 'Half Big')

    plt.savefig(fig_path / 'fig_example_{}.png'.format(index_b),dpi=300)
    plt.close()

def plot_fields_all(POT, p_2, p_4, p_8, p_16, p_32, index_b, fig_path, not_orig):
    """
    Useful plot functions that plot the fields: 1/1, 1/2, 1/4, 1/8, 1/16, 1/32 
    The not_orig flag determines if the interpolated fields have the original size (i.e. for the 1/4 field, nnx, nny) ==> True 
    Or the inteprolated size (i.e. for the 1/4 field, nnx/4, nny/4) ==> False
    """

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


def inside_plot(fig,axis,n,n_f, ones, log_t, diff, title, limit, axes_max):
    """
    Contourf function for the 2D FFT fields.
    """
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
    """
    Plots the 2D FFT fields, or the FFT of the difference between the interpolated and the original field (diff=True). 
    Optionally in log scale (log_t = True).
    """
    axes_max = 750
    limit = True
    ones = np.ones_like(n)

    # Begin Figure
    fig, ax = plt.subplots(figsize=(50, 10), nrows=1, ncols=6)

    inside_plot( fig, ax[0], n, n_f, ones, log_t, diff, 'Original Scale', limit, axes_max)
    inside_plot( fig, ax[1], n2, n2_f, ones, log_t, diff, 'N/2 Scale', limit, axes_max)
    inside_plot( fig, ax[2], n4, n4_f, ones, log_t, diff, 'N/4 Scale', limit, axes_max)
    inside_plot( fig, ax[3], n8, n8_f, ones, log_t, diff, 'N/8 Scale', limit, axes_max)
    inside_plot( fig, ax[4], n16, n16_f, ones, log_t, diff, 'N/16 Scale', limit, axes_max)
    inside_plot( fig, ax[5], n32, n32_f, ones, log_t, diff, 'N/32 Scale', limit, axes_max)

    # Save and close depending on options
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

