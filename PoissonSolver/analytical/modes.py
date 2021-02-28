import os
import numpy as np
import matplotlib.pyplot as plt

from PlasmaNet.common.plot import plot_ax_scalar
from PlasmaNet.common.utils import create_dir

def mode(X, Y, Lx, Ly, n, m):
    return np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly)

if __name__=='__main__':
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
    fig, axes = plt.subplots(ncols=N, nrows=M, figsize=(20, 20))
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            plot_ax_scalar(fig, axes[n - 1][m - 1], X, Y, 
            mode(X, Y, Lx, Ly, n, m), 'n = %d m = %d' % (n, m), cbar=False)
    plt.tight_layout()
    plt.savefig(fig_dir + 'modes_%d_%d' % (N, M))