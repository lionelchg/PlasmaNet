########################################################################################################################
#                                                                                                                      #
#                                                    Plot functions                                                    #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_ax_scalar(fig, ax, X, Y, field, title):
    levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap='RdBu')
    fig.colorbar(cs1, ax=ax, pad=0.05, fraction=0.1, aspect=7)
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_scalar(X, Y, u, res, dtsum, number, fig_dir):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
    plot_ax_scalar(fig, axes[0], X, Y, u, "Scalar")
    plot_ax_scalar(fig, axes[1], X, Y, res, "Residual")
    plt.tight_layout()
    plt.figtext(0.85, 0.07, '$t =$%.2e s' % dtsum, fontsize=12)
    plt.savefig(fig_dir + 'instant_%04d' % number, bbox_inches='tight')
