import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PlasmaNet.common.plot import plot_ax_scalar
from PlasmaNet.common.utils import create_dir


def dirichlet_mode(x:np.ndarray, Lx:float, n:int) -> np.ndarray:
    """ Mode for 1D full Dirichlet Poisson problem

    :param x: direction vector
    :type x: np.ndarray
    :param Lx: length of the domain in the direction
    :type Lx: float
    :param n: mode number
    :type n: int
    :return: mode of number n
    :rtype: np.ndarray
    """
    return np.sin(n * np.pi * x / Lx)

def neumann_mode(x:np.ndarray, Lx:float, n:int) -> np.ndarray:
    """ Mode for 1D full Neumann Poisson problem

    :param x: direction vector
    :type x: np.ndarray
    :param Lx: length of the domain in the direction
    :type Lx: float
    :param n: mode number
    :type n: int
    :return: mode of number n
    :rtype: np.ndarray
    """
    return np.cos(n * np.pi * x / Lx)

def mixed_mode_ldrn(x:np.ndarray, Lx:float, n:int) -> np.ndarray:
    """ Mode for 1D left Dirichlet - right Neumann Poisson problem

    :param x: direction vector
    :type x: np.ndarray
    :param Lx: length of the domain in the direction
    :type Lx: float
    :param n: mode number
    :type n: int
    :return: mode of number n
    :rtype: np.ndarray
    """
    return np.sin((n + 0.5) * np.pi * x / Lx)

def mixed_mode_lnrd(x:np.ndarray, Lx:float, n:int) -> np.ndarray:
    """ Mode for 1D left Neumann - right Dirichlet Poisson problem

    :param x: direction vector
    :type x: np.ndarray
    :param Lx: length of the domain in the direction
    :type Lx: float
    :param n: mode number
    :type n: int
    :return: mode of number n
    :rtype: np.ndarray
    """
    return np.cos((n + 0.5) * np.pi * x / Lx)

if __name__=='__main__':
    sns.set_context('notebook', font_scale=0.9, rc={"lines.linewidth": 1.5})
    fig_dir = 'figures/modes/'
    create_dir(fig_dir)

    n_points = 101
    xmin, xmax = 0, 0.01
    ymin, ymax = 0, 0.01
    Lx, Ly = xmax - xmin, ymax - ymin
    dx, dy = (xmax - xmin) / (n_points - 1), (ymax - ymin) / (n_points - 1)
    x, y = np.linspace(xmin, xmax, n_points), np.linspace(ymin, ymax, n_points)

    X, Y = np.meshgrid(x, y)
    N, M = 3, 3

    # Full dirichlet modes
    fig, axes = plt.subplots(ncols=N, nrows=M, figsize=(10, 10))
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            plot_ax_scalar(fig, axes[n - 1][m - 1], X, Y, 
            dirichlet_mode(X, Lx, n) * dirichlet_mode(Y, Ly, m), 'n = %d m = %d' % (n, m), cbar=False)
    plt.tight_layout()
    plt.savefig(fig_dir + 'dirichlet_%d_%d' % (N, M))
    plt.close()

    # Full neumann modes
    fig, axes = plt.subplots(ncols=N, nrows=M, figsize=(10, 10))
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            plot_ax_scalar(fig, axes[n - 1][m - 1], X, Y, 
            neumann_mode(X, Lx, n) * neumann_mode(Y, Ly, m), 'n = %d m = %d' % (n, m), cbar=False)
    plt.tight_layout()
    plt.savefig(fig_dir + 'neumann_%d_%d' % (N, M))
    plt.close()

    # Modes for dirichlet at left and bottom, neumann at right and top boundaries
    fig, axes = plt.subplots(ncols=N, nrows=M, figsize=(10, 10))
    for n in range(N):
        for m in range(M):
            plot_ax_scalar(fig, axes[n - 1][m - 1], X, Y, 
            mixed_mode_ldrn(X, Lx, n) * mixed_mode_ldrn(Y, Ly, m), 'n = %d m = %d' % (n, m), cbar=False)
    plt.tight_layout()
    plt.savefig(fig_dir + 'mixed_ldrn_%d_%d' % (N, M))
    plt.close()

    # Modes for Neumann at left and bottom, Dirichlet at right and top boundaries
    fig, axes = plt.subplots(ncols=N, nrows=M, figsize=(10, 10))
    for n in range(N):
        for m in range(M):
            plot_ax_scalar(fig, axes[n - 1][m - 1], X, Y, 
            mixed_mode_lnrd(X, Lx, n) * mixed_mode_lnrd(Y, Ly, m), 'n = %d m = %d' % (n, m), cbar=False)
    plt.tight_layout()
    plt.savefig(fig_dir + 'mixed_lnrd_%d_%d' % (N, M))
    plt.close()

