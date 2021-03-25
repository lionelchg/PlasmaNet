########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import copy

from scipy import sparse
from scipy.sparse.linalg import spsolve
from numba import njit

from ...common.utils import create_dir
from ..base.base_plot import plot_ax_scalar, plot_ax_scalar_1D
from ...poissonsolver.linsystem import matrix_axisym, impose_dirichlet


lambda_j_three = np.array([0.0553, 0.1460,0.89]) * 1.0e2
A_j_three = np.array([1.986e-4, 0.0051, 0.4886]) * (1.0e2)**2
lambda_j_two = np.array([0.0974, 0.5877]) * 1.0e2
A_j_two = np.array([0.0021, 0.1775]) * (1.0e2)**2

coef_p = 0.038

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)


def plot_Sph(X, R, dx, dr, Sph, nx, nr, figname):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 10))
    plot_ax_scalar(fig, axes[0], X, R, Sph, 'Sph 2D', cmap_scale='log', geom='xr', field_ticks=[1e23, 1e26, 1e29])
    plot_ax_scalar_1D(fig, axes[1], X, [0, 0.05, 0.1], Sph, "Sph 1D cuts", yscale='log', ylim=[1e23, 1e29])
    plt.savefig(figname)


def plot_Sph_irate(X, R, dx, dr, Sph, irate, nx, nr, figname):
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
    plot_ax_scalar(fig, axes[0], X, R, irate, 'Ionization rate', geom='xr')
    plot_ax_scalar(fig, axes[1], X, R, Sph, 'Sph', geom='xr')
    plt.savefig(figname, bbox_inches='tight')


@njit(cache=True)
def photo_coeff(E_p):
    # In Zheleznyak paper, the tabulation is done with E/p in V/cm * mmHg
    E_p = E_p * 133.32 / 100
    if E_p < 30:
        pcoeff = 5.e-2
    elif 30 <= E_p < 50:
        pcoeff = 0.07 / 20 * (E_p - 30) + 5.e-2
    elif 50 <= E_p < 100:
        pcoeff = 0.12 - 4e-2 / 50 * (E_p - 50)
    else:
        pcoeff = 0.08 - 2e-2 / 100 * (E_p - 100)
    return pcoeff


if __name__ == '__main__':
    fig_dir = 'figures/photo/'
    create_dir(fig_dir)
    xmin, xmax, nx = 0, 2e-3, 252
    rmin, rmax, nr = 0, 2e-3, 252
    dx, dr = (xmax - xmin) / (nx - 1), (rmax - rmin) / (nr - 1)
    x, r = np.linspace(xmin, xmax, nx), np.linspace(rmin, rmax, nr)

    X, R = np.meshgrid(x, r)

    scale = dx * dr

    # creating the rhs
    I0 = 3.5e28
    sigma_x, sigma_r = 1e-4, 1e-4
    x0, r0 = 1e-3, 0
    rhs = np.zeros(nx * nr)

    # params
    Torr = 133.32
    pO2 = 150

    # interior rhs
    I = gaussian(X.reshape(-1), R.reshape(-1), I0, x0, r0, sigma_x, sigma_r)

    # Boundary conditions
    up = np.zeros_like(x)
    down = np.zeros_like(x)
    left = np.zeros_like(r)
    right = np.zeros_like(r)

    bcs = {'left':left, 'right':right, 'top':up}
    
    Sph = np.zeros_like(X)
    for i in range(2):
        # Axisymmetric resolution
        R_nodes = copy.deepcopy(R)
        R_nodes[0] = dr / 4
        A = matrix_axisym(dx, dr, nx, nr, R_nodes, (lambda_j_two[i] * pO2)**2, scale)
        rhs = - I * A_j_two[i] * pO2**2 * scale
        impose_dirichlet(rhs, nx, nr, bcs)
        Sph += spsolve(A, rhs).reshape(nr, nx)

    # Plots
    Sph = Sph.reshape(nr, nx)
    plot_Sph(X, R, dx, dr, Sph, nx, nr, fig_dir + 'Sph_two')
