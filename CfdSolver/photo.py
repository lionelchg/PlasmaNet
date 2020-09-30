########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import copy

from scipy import sparse
from scipy.sparse.linalg import spsolve
from plot import plot_ax_scalar, plot_ax_scalar_1D
from numba import njit

fig_dir = 'figures/photo/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

lambda_j_three = np.array([0.0553, 0.1460,0.89]) * 1.0e2
A_j_three = np.array([1.986e-4, 0.0051, 0.4886]) * (1.0e2)**2
lambda_j_two = np.array([0.0974, 0.5877]) * 1.0e2
A_j_two = np.array([0.0021, 0.1775]) * (1.0e2)**2

coef_p = 0.038

def photo_axisym(dx, dr, nx, nr, R, coeff, scale):
    diags = np.zeros((5, nx * nr))

    r = R.reshape(-1)

    # Filling the diagonals, first the down neumann bc,: the dirichlet bc and finally the interior nodes
    for i in range(nx * nr):
        if 0 < i < nx - 1:
            diags[0, i] = - (2 / dx**2 + 4 / dr**2 + coeff) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = 4 / dr**2 * scale
        elif i >= (nr - 1) * nx or i % nx == 0 or i % nx == nx - 1:
            diags[0, i] = 1
            diags[1, min(i + 1, nx * nr - 1)] = 0
            diags[2, max(i - 1, 0)] = 0
            diags[3, min(i + nx, nx * nr - 1)] = 0
            diags[4, max(i - nx, 0)] = 0
        else:
            diags[0, i] = - (2 / dx**2 + 2 / dr**2 + coeff) * scale
            diags[1, i + 1] = 1 / dx**2 * scale
            diags[2, i - 1] = 1 / dx**2 * scale
            diags[3, i + nx] = (1 + dr / (2 * r[i])) / dr**2 * scale
            diags[4, i - nx] = (1 - dr / (2 * r[i])) / dr**2 * scale

    # Creating the matrix
    return sparse.csc_matrix(
        sparse.dia_matrix((diags, [0, 1, -1, nx, -nx]), shape=(nx * nr, nx * nr)))

def dirichlet_bc_axi(rhs, nx, nr, up, left, right):
    # filling of the three dirichlet boundaries for axisymmetric test case
    rhs[nx * (nr - 1):] = up
    rhs[:nx * (nr - 1) + 1:nx] = left
    rhs[nx - 1::nx] = right
    # mean approximation in case the potential is not continuous across boundaries
    rhs[-nx] = 0.5 * (left[-1] + up[0])
    rhs[-1] = 0.5 * (right[-1] + up[-1])

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
    if (E_p < 30):
        pcoeff = 5.e-2
    elif (E_p >= 30 and E_p < 50):
        pcoeff = 0.07 / 20 * (E_p - 30) + 5.e-2
    elif (E_p >= 50 and E_p < 100):
        pcoeff = 0.12 - 4e-2 / 50 * (E_p - 50)
    else:
        pcoeff = 0.08 - 2e-2 / 100 * (E_p - 100)
    return pcoeff

if __name__ == '__main__':
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
    
    Sph = np.zeros_like(X)
    for i in range(2):
        # Axisymmetric resolution
        R_nodes = copy.deepcopy(R)
        R_nodes[0] = dr / 4
        A = photo_axisym(dx, dr, nx, nr, R_nodes, (lambda_j_two[i] * pO2)**2, scale)
        rhs = - I * A_j_two[i] * pO2**2 * scale
        dirichlet_bc_axi(rhs, nx, nr, up, left, right)
        Sph += spsolve(A, rhs).reshape(nr, nx)

    # Plots
    Sph = Sph.reshape(nr, nx)
    plot_Sph(X, R, dx, dr, Sph, nx, nr, fig_dir + 'Sph_two')
