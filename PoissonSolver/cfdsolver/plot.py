########################################################################################################################
#                                                                                                                      #
#                                                    Plot functions                                                    #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return np.ceil(n * multiplier) / multiplier

def plot_ax_scalar(fig, ax, X, Y, field, title, cmap_scale=None, cmap='RdBu', aspect=7):
    if cmap_scale == 'log':
        cmap = 'Blues'
        field_ticks = [1e14, 1e17, 1e20]
        pows = np.log10(np.array(field_ticks)).astype(int)
        levels = np.logspace(pows[0], pows[-1], 100, endpoint=True)
        field = np.maximum(field, field_ticks[0])
        field = np.minimum(field, field_ticks[-1])
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap, norm=LogNorm())
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=0.1, aspect=aspect, ticks=field_ticks)
    else:
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
        cs1 = ax.contourf(X, Y, field, levels, cmap=cmap)
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=0.1, aspect=aspect)
    ax.set_aspect("equal")
    ax.set_title(title)

def plot_scalar(X, Y, u, res, dtsum, number, fig_dir):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    plot_ax_scalar(fig, axes[0], X, Y, u, "Scalar")
    plot_ax_scalar(fig, axes[1], X, Y, res, "Residual")
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'instant_%04d' % number, bbox_inches='tight')

def plot_streamer(X, Y, ne, rese, nionp, resp, nn, resn, dtsum, number, fig_dir):
    aspect = 4
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    plot_ax_scalar(fig, axes[0][0], X, Y, ne, "$n_e$", cmap_scale='log', aspect=aspect)
    plot_ax_scalar(fig, axes[0][1], X, Y, rese, "$r_e$", aspect=aspect)
    plot_ax_scalar(fig, axes[1][0], X, Y, nionp, "$n_p$", cmap_scale='log', aspect=aspect)
    plot_ax_scalar(fig, axes[1][1], X, Y, resp, "$r_p$", aspect=aspect)
    plot_ax_scalar(fig, axes[2][0], X, Y, nn, "$n_n$", cmap_scale='log', aspect=aspect)
    plot_ax_scalar(fig, axes[2][1], X, Y, resn, "$r_n$", aspect=aspect)
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'dens_instant_%04d' % number, bbox_inches='tight')