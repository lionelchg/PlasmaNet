import numpy as np
import matplotlib.pyplot as plt

from PlasmaNet.common.operators_numpy import lapl, grad
from PlasmaNet.common.plot import plot_ax_scalar, plot_ax_vector_arrow


def plot_potential(X, Y, dx, dy, potential, n_points, figname, figtitle=None):
    # 1D vector
    x = X[0, :]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 14))

    plot_ax_scalar(fig, axes[0][0], X, Y, potential, 'Potential')
    plot_ax_trial_1D(axes[0][1], x, potential, n_points, '1D cuts')

    E = grad(potential, dx, dy, n_points, n_points)
    normE = np.sqrt(E[0]**2 + E[1]**2)
    plot_ax_vector_arrow(fig, axes[1][0], X, Y, E, 'Electric field')
    plot_ax_trial_1D(axes[1][1], x, normE, n_points, '1D cuts')

    lapl_trial = lapl(potential, dx, dy, n_points, n_points)
    plot_ax_scalar(fig, axes[2, 0], X, Y, - lapl_trial, '- Laplacian')
    plot_ax_trial_1D(axes[2][1], x, -  lapl_trial, n_points, '1D cuts')

    if figtitle is not None:
        plt.suptitle(figtitle)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(figname, bbox_inches='tight')


def plot_ax_trial_1D(ax, x, function, n_points, title):
    direction = 'y'
    list_cut = [0, 0.25, 0.5]
    for cut_pos in list_cut:
        n = int(cut_pos * (n_points - 1))
        ax.plot(x, function[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def plot_ax_3D(fig, index, X, Y, Z, title):
    ax = fig.add_subplot(2, 2, index, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.5)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Power')
    ax.set_zlabel('Loss')
    ax.set_title(title)
    return ax
