########################################################################################################################
#                                                                                                                      #
#                                    Scalar transport equations related routines                                       #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from numba import njit
import logging

import PlasmaNet.common.profiles as pf
from ..base.base_sim import BaseSim
from .boundary import outlet_x, outlet_y, full_perio, perio_x, perio_y
from ...common.operators_numpy import grad
from ...common.plot import plot_ax_scalar, plot_ax_scalar_1D


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
        self.res = np.zeros_like(self.X)
        self.u = getattr(pf, config['params']['init_func'][0])(self.X, self.Y, *config['params']['init_func'][1])

    def print_init(self):
        """ Print header to sum up the parameters. """
        logging.info(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        logging.info(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        logging.info(f'Transport: a = {self.max_speed:.2e} -- D = {self.max_D:.2e}')
        logging.info(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e} -- CFL = {self.cfl:.2e} -- Fourier = {self.fourier:.2e} -- Timestep = {self.dt:.2e}')
        logging.info('------------------------------------')
        logging.info('Start of simulation')
        logging.info('------------------------------------')
        logging.info('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    def compute_dflux(self):
        """ Compute the diffusive flux: D * grad(u) """
        self.diff_flux = self.D * grad(self.u, self.dx, self.dy, self.nnx, self.nny)

    def compute_flux(self):
        """ Wrapper for jitted compute_flux function. """
        res = self.res
        a = self.a
        u = self.u
        diff_flux = self.diff_flux
        sij = self.sij
        compute_flux(res, a, u, diff_flux, sij, self.ncx, self.ncy, r=self.R_nodes)

    def impose_bc(self):
        """ Impose boundary conditions specified in the config file. """
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
                outlet_y(res, a, u, diff_flux, self.dx, -1, r=np.max(self.Y))
                outlet_x(res, a, u, diff_flux, self.dy, 0, r=self.Y)
                outlet_x(res, a, u, diff_flux, self.dy, -1, r=self.Y)

    def update_res(self):
        """ Update residual. """
        self.u -= self.res * self.dt / self.voln

    def plot(self):
        """ Execute control plots. """
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.u, "Scalar", geom=self.geom)
        plot_ax_scalar_1D(fig, axes[1], self.X, [0.1, 0.25, 0.5], self.u, "Scalar")
        plt.tight_layout()
        fig.suptitle(f'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(self.fig_dir / f'instant_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)

    @classmethod
    def run(cls, config):
        """ Main function containing initialisation, temporal loop and outputs. Takes a config dict as input. """

        sim = cls(config)

        # Print header to sum up the parameters
        sim.print_init()

        # Iterations
        for it in range(1, sim.nit + 1):
            sim.dtsum += sim.dt

            # Calculation of diffusive flux
            sim.compute_dflux()

            # Update of the residual to zero
            sim.res[:] = 0

            # Loop on the cells to compute the interior flux and update residuals
            sim.compute_flux()

            # Impose boundary conditions
            sim.impose_bc()

            # Update residuals u -= res * dt / voln
            sim.update_res()

            # Post processing (printing and plotting)
            sim.postproc(it)


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
    and second order centered scheme for diffusion. """
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

def main():
    args = argparse.ArgumentParser(description='ScalarTransport run')
    args.add_argument('-c', '--config', type=str,
                        help='Config filename', required=True)
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    ScalarTransport.run(cfg)

if __name__ == '__main__':
    main()