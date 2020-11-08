########################################################################################################################
#                                                                                                                      #
#                                    Scalar transport equations related routines                                       #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import copy
from numba import njit
from ..base.base_plot import plot_ax_scalar, plot_ax_scalar_1D
from ..base.basesim import BaseSim
from ..base.operators import grad
from boundary import outlet_x, outlet_y, full_perio, perio_x, perio_y

class ScalarTransport(BaseSim):
    def __init__(self, config):
        super().__init__(config)

        # Convection speed
        self.a = np.zeros((self.ndim, self.nny, self.nnx))
        self.a[0, :, :] = config['transport']['convection_x']
        self.a[1, :, :] = config['transport']['convection_y']
        norm_a = np.sqrt(self.a[0, :, :] ** 2 + self.a[1, :, :] ** 2)
        self.max_speed = np.max(norm_a)

        # Diffusion coefficient
        self.D = np.zeros_like(self.X)
        self.D[:] = config['transport']['diffusion']
        self.max_D = np.max(self.D)
    
        self.fourier = config['params']['fourier']
        self.dt = min(self.cfl * self.dx / self.max_speed, self.fourier * self.dx ** 2 / self.max_D)

        # Scalar and Residual declaration
        self.u, self.res = np.zeros_like(self.X), np.zeros_like(self.X)

    def print_init(self):
        # Print header to sum up the parameters
        print(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        print(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        print(f'Transport: a = {self.max_speed:.2e} -- D = {self.max_D:.2e}')
        print(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e} -- CFL = {self.cfl:.2e} -- Fourier = {self.fourier:.2e} -- Timestep = {self.dt:.2e}')
        print('------------------------------------')
        print('Start of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    def compute_dflux(self):
        self.diff_flux = self.D * grad(self.u, self.dx, self.dy, self.nnx, self.nny)

    def compute_flux(self):
        res = self.res
        a = self.a
        u = self.u
        diff_flux = self.diff_flux
        sij = self.sij
        compute_flux(res, a, u, diff_flux, sij, self.ncx, self.ncy, r=self.R_nodes)
    
    def impose_bc(self):
        """ Impose boundary conditions specified in the config file """
        res = self.res
        a = self.a
        u = self.u
        diff_flux = self.diff_flux
        # Boundary conditions
        if self.BC == 'full_perio':
            full_perio(res)
        elif self.BC == 'perio_x':
            perio_x(res)
            outlet_y(res, a, u, diff_flux, self.dx, 0)
            outlet_y(res, a, u, diff_flux, self.dx, -1)
        elif self.BC == 'perio_y':
            perio_y(res)
            outlet_x(res, a, u, diff_flux, self.dy, 0)
            outlet_x(res, a, u, diff_flux, self.dy, -1)
        elif self.BC == 'full_out':
            if self.geom == 'xy':
                outlet_y(res, a, u, diff_flux, self.dx, 0)
                outlet_y(res, a, u, diff_flux, self.dx, -1)
                outlet_x(res, a, u, diff_flux, self.dy, 0)
                outlet_x(res, a, u, diff_flux, self.dy, -1)
            elif self.geom == 'xr':
                outlet_y(res, a, u, diff_flux, self.dx, -1, r=np.max(Y))
                outlet_x(res, a, u, diff_flux, self.dy, 0, r=Y)
                outlet_x(res, a, u, diff_flux, self.dy, -1, r=Y)
    
    def update_res(self):
        self.u -= self.res * self.dt / self.voln
    
    def plot(self):
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.u, "Scalar", geom=self.geom)
        plot_ax_scalar_1D(fig, axes[1], self.X, [0.1, 0.25, 0.5], self.u, "Scalar")
        plt.tight_layout()
        fig.suptitle(f'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(self.fig_dir + 'instant_%04d' % self.number, bbox_inches='tight')
        plt.close(fig)


@njit(cache=True)
def compute_flux(res, a, u, diff_flux, sij, ncx, ncy, r=None):
    """ Iterate over cells (their edges) to compute the flux and residual. """
    for i in range(ncx):
        for j in range(ncy):
            if r is not None:
                edge_flux(res, a, u, diff_flux, sij * (0.75 * r[j] + 0.25 * r[j + 1]), i, j, i + 1, j, 0)
                edge_flux(res, a, u, diff_flux, sij * (0.25 * r[j] + 0.75 * r[j + 1]), i, j + 1, i + 1, j + 1, 0)
                edge_flux(res, a, u, diff_flux, sij * (r[j] + r[j + 1]) / 2, i, j, i, j + 1, 1)
                edge_flux(res, a, u, diff_flux, sij * (r[j] + r[j + 1]) / 2, i + 1, j, i + 1, j + 1, 1)
            else:
                edge_flux(res, a, u, diff_flux, sij, i, j, i + 1, j, 0)
                edge_flux(res, a, u, diff_flux, sij, i, j + 1, i + 1, j + 1, 0)
                edge_flux(res, a, u, diff_flux, sij, i, j, i, j + 1, 1)
                edge_flux(res, a, u, diff_flux, sij, i + 1, j, i + 1, j + 1, 1)


@njit(cache=True)
def edge_flux(res, a, u, diff_flux, sij, i1, j1, i2, j2, dim):
    """ Convection-diffusion flux. Implemented with 1st order upwind scheme for convection
    and second order centered scheme for diffusion """
    # Convective flux
    scalar_product = 0.5 * (a[dim, j1, i1] + a[dim, j2, i2]) * sij[dim]
    if scalar_product >= 0:
        flux = scalar_product * u[j1, i1]
    else:
        flux = scalar_product * u[j2, i2]
    # Diffusive flux
    flux -= 0.5 * (diff_flux[dim, j1, i1] + diff_flux[dim, j2, i2]) * sij[dim]
    res[j1, i1] += flux
    res[j2, i2] -= flux
