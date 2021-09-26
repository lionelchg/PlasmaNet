########################################################################################################################
#                                                                                                                      #
#                                          Euler equations related routines                                            #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import copy
import yaml
import argparse
from numba import njit
import logging

from ...common.plot import plot_ax_scalar, plot_ax_scalar_1D
from ..base.base_sim import BaseSim

class Euler(BaseSim):
    def __init__(self, config):
        super().__init__(config)

        # Scalar and residual declaration - Mixture parameters
        self.gamma = 7 / 5
        self.neqs = 4
        self.W = 0.029  # kg/mol
        self.r = co.R / self.W
        self.U = np.zeros((self.neqs, self.nny, self.nnx))
        self.U_c = np.zeros((self.neqs, self.ncy, self.ncx))
        self.F = np.zeros((self.neqs, self.ndim, self.nny, self.nnx))
        self.res = np.zeros_like(self.U)
        self.res_c = np.zeros((self.neqs, self.ncy, self.ncx))
        self.press, self.Tgas = np.zeros_like(self.X), np.zeros_like(self.X)
        self.dF_c = np.zeros((self.neqs, self.ndim))

    def print_init(self):
        """ Print header to sum up the parameters. """
        logging.info(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        logging.info(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        logging.info(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e} -- CFL = {self.cfl:.2e}')
        logging.info('------------------------------------')
        logging.info('Start of simulation')
        logging.info('------------------------------------')
        logging.info('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    def compute_flux(self):
        """ Compute the 2D flux of the Euler equations as well as pressure and
        temperature with wrapper around numba routine (optimized for speed). """
        F = self.F
        U = self.U
        press = self.press
        Tgas = self.Tgas
        compute_flux(U, self.gamma, self.r, F, press, Tgas)

    def compute_timestep(self):
        """ Compute the time step with wraper around numba routine. """
        press = self.press
        U = self.U
        self.dt = compute_timestep(self.cfl, self.dx, U, press, self.gamma)
        self.dtsum += self.dt

    def compute_res(self):
        """ Compute residuals for the Lax-Wendroff scheme in a cell-vertex fashion
        (wrapper around numba routine). """
        U = self.U
        U_c = self.U_c
        res_c = self.res_c
        res = self.res
        F = self.F
        dF_c = self.dF_c
        snc = self.snc

        compute_res(U, F, self.dt, self.volc, self.snc, self.ncx, self.ncy,
                    self.gamma, self.ndim, self.nvert, res, res_c, U_c)

    def impose_bc_euler(self):
        """ Full periodic conditions in the 4 directions. """
        res = self.res
        if self.BC == 'full_perio':
            # Corner (only for full periodic)
            res[:, 0, 0] = res[:, 0, 0] + res[:, 0, -1] + res[:, -1, 0] + res[:, -1, -1]
            res[:, 0, -1], res[:, -1, 0], res[:, -1, -1] = res[:, 0, 0], res[:, 0, 0], res[:, 0, 0]

            # Periodic boundary conditions - Sides
            res[:, 1:-1, 0] += res[:, 1:-1, -1]
            res[:, 1:-1, -1] = copy.deepcopy(res[:, 1:-1, 0])
            res[:, 0, 1:-1] += res[:, -1, 1:-1]
            res[:, -1, 1:-1] = copy.deepcopy(res[:, 0, 1:-1])

    def update_res(self):
        """ Apply residuals. """
        self.U -= self.dt / self.voln * self.res

    def plot(self):
        """ 2D maps and 1D cuts at different y of the primitive variables. """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.U[0], r"$\rho$", geom='xy')
        press = (self.gamma - 1) * (self.U[3] - (self.U[1]**2 + self.U[2]**2) / 2 / self.U[0])
        plot_ax_scalar(fig, axes[0][1], self.X, self.Y, press, "$P$", geom='xy')
        plot_ax_scalar(fig, axes[1][0], self.X, self.Y, self.U[1] / self.U[0], "$u$", geom='xy')
        plot_ax_scalar(fig, axes[1][1], self.X, self.Y, self.U[2] / self.U[0], "$v$", geom='xy')
        plt.suptitle(rf'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(self.fig_dir / f'2D_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plot_ax_scalar_1D(fig, axes[0][0], self.X, [0.25, 0.5], self.U[0], r"$\rho$")
        press = (self.gamma - 1) * (self.U[3] - (self.U[1]**2 + self.U[2]**2) / 2 / self.U[0])
        plot_ax_scalar_1D(fig, axes[0][1], self.X, [0.25, 0.5], press, "$P$")
        plot_ax_scalar_1D(fig, axes[1][0], self.X, [0.25, 0.5], self.U[1] / self.U[0], "$u$")
        plot_ax_scalar_1D(fig, axes[1][1], self.X, [0.25, 0.5], self.U[2] / self.U[0], "$v$")
        plt.suptitle(rf'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(self.fig_dir / f'1D_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)

    @classmethod
    def run(cls, config):
        """ Main function containing initialization, temporal loop and outputs. Takes a config dict as input. """

        sim = cls(config)

        # Convective vortex parameters and initialization
        x0, y0 = 0, 0
        u0, v0 = 2, 0
        rho0, p0 = 1, 1 / sim.gamma
        alpha, K = 1, 5
        a0 = np.sqrt(sim.gamma * p0 / rho0)
        T0 = 1 / sim.gamma / sim.r
        covo(sim.X, sim.Y, x0, y0, u0, v0, rho0, p0, T0, alpha, K, sim.gamma, sim.r, 0, sim.U)

        # Print header to sum up the parameters
        if sim.verbose:
            sim.print_init()

        # Iterations
        for it in range(1, sim.nit + 1):
            # Update of the residual to zero
            sim.res[:], sim.res_c[:] = 0, 0

            # Compute euler fluxes
            sim.compute_flux()

            # Calculate timestep based on the maximum value of u + ss
            sim.compute_timestep()

            # Compute residuals in cell-vertex method
            sim.compute_res()

            # boundary conditions
            sim.impose_bc_euler()

            # Apply residual
            sim.update_res()

            # Post processing
            sim.postproc(it)


def covo(x, y, x0, y0, u0, v0, rho0, p0, T0, alpha, K, gamma, r, t, U):
    """ Initialize isentropic convective vortex as given by idolikecfd chap 7. """
    xbar = x - x0 - u0 * t
    ybar = y - y0 - v0 * t
    rbar = np.sqrt(xbar**2 + ybar**2)
    u = u0 - K / (2 * np.pi) * ybar * np.exp(alpha * (1 - rbar**2) / 2)
    v = v0 + K / (2 * np.pi) * xbar * np.exp(alpha * (1 - rbar**2) / 2)
    T = T0 - K**2 * (gamma - 1) / (8 * alpha * np.pi**2 * gamma * r) * np.exp(alpha * (1 - rbar**2))
    rho = rho0 * (T / T0)**(1 / (gamma - 1))
    p = p0 * (T / T0)**(gamma / (gamma - 1))
    # Define conservative variables
    U[0] = rho  # density
    U[1] = rho * u  # momentum along x
    U[2] = rho * v  # momentum along y
    U[3] = rho / 2 * (u**2 + v**2) + p / (gamma - 1)  # total energy with closure on internal energy


@njit(cache=True)
def compute_timestep(cfl, dx, U, press, gamma):
    """ Compute timestep CFL condition. """
    speed = np.sqrt(gamma * press / U[0]) + np.sqrt((U[1]**2 + U[2]**2) / U[0])
    dt = cfl * dx / np.max(speed[:])
    return dt


@njit(cache=True)
def compute_flux(U, gamma, r, F, press, Tgas):
    """ Compute the 2D flux of the Euler equations
    as well as pressure and temperature. """
    press = (gamma - 1) * (U[3] - (U[1]**2 + U[2]**2) / 2 / U[0])
    Tgas = press / U[0] / r
    # rhou - rhov
    F[0, 0] = U[1]
    F[0, 1] = U[2]
    # rho u^2 + p - rho u v
    F[1, 0] = U[1]**2 / U[0] + press
    F[1, 1] = U[1] * U[2] / U[0]
    # rho u^2 + p - rho u v
    F[2, 0] = U[1] * U[2] / U[0]
    F[2, 1] = U[2]**2 / U[0] + press
    # u(rho E + p) - v(rho E + p)
    F[3, 0] = U[1] / U[0] * (U[3] + press)
    F[3, 1] = U[2] / U[0] * (U[3] + press)


@njit(cache=True)
def compute_res(U, F, dt, volc, snc, ncx, ncy, gamma, ndim, nvert, res, res_c, U_c):
    """ Compute residuals for the Lax-Wendroff scheme in a cell-vertex fashion. """
    dF_c = np.zeros((4, 2))
    for i in range(ncx):
        for j in range(ncy):
            U_c[:, j, i] = 0.25 * (U[:, j, i] + U[:, j + 1, i]
                                   + U[:, j, i + 1] + U[:, j + 1, i + 1])
            res_c[:, j, i] = - (F[:, 0, j, i] * snc[0, 0] + F[:, 1, j, i] * snc[0, 1]
                                + F[:, 0, j, i + 1] * snc[1, 0] + F[:, 1, j, i + 1] * snc[1, 1]
                                + F[:, 0, j + 1, i + 1] * snc[2, 0] + F[:, 1, j + 1, i + 1] * snc[2, 1]
                                + F[:, 0, j + 1, i] * snc[3, 0] + F[:, 1, j + 1, i] * snc[3, 1])
            # Add central term
            res[:, j, i] += res_c[:, j, i] / nvert
            res[:, j, i + 1] += res_c[:, j, i] / nvert
            res[:, j + 1, i + 1] += res_c[:, j, i] / nvert
            res[:, j + 1, i] += res_c[:, j, i] / nvert
    res_c *= dt / volc

    # Add second order diffusion term
    for i in range(ncx):
        for j in range(ncy):
            press_c = (gamma - 1) * (U_c[3, j, i] -
                                     (U[1, j, i]**2 + U[2, j, i]**2) / 2 / U[0, j, i])
            rho = U_c[0, j, i]
            u = U_c[1, j, i] / rho
            v = U_c[2, j, i] / rho
            H = U_c[3, j, i] / rho + press_c / rho
            beta = gamma - 1
            alpha = 0.5 * (u**2 + v**2)

            drho = res_c[0, j, i]
            drhou = res_c[1, j, i]
            drhov = res_c[2, j, i]
            drhoE = res_c[3, j, i]

            rhodu = drhou - u * drho
            rhodv = drhov - v * drho
            drhouu = u * drhou + u * rhodu
            drhovv = v * drhov + v * rhodv
            drhouv = u * drhov + v * rhodu

            dP = beta * (drhoE - u * drhou - v * drhov) + beta * alpha * drho

            drhoH = drhoE + dP
            drhoHu = H * rhodu + u * drhoH
            drhoHv = H * rhodv + v * drhoH

            dF_c[0, 0] = drhou
            dF_c[1, 0] = drhouu + dP
            dF_c[2, 0] = drhouv
            dF_c[3, 0] = drhoHu

            dF_c[0, 1] = drhov
            dF_c[1, 1] = drhouv
            dF_c[2, 1] = drhovv + dP
            dF_c[3, 1] = drhoHv

            res[:, j, i] -= 0.5 * (dF_c[:, 0] * snc[0, 0] + dF_c[:, 1] * snc[0, 1])
            res[:, j, i + 1] -= 0.5 * (dF_c[:, 0] * snc[1, 0] + dF_c[:, 1] * snc[1, 1])
            res[:, j + 1, i + 1] -= 0.5 * (dF_c[:, 0] * snc[2, 0] + dF_c[:, 1] * snc[2, 1])
            res[:, j + 1, i] -= 0.5 * (dF_c[:, 0] * snc[3, 0] + dF_c[:, 1] * snc[3, 1])


def main():
    args = argparse.ArgumentParser(description='Euler run')
    args.add_argument('-c', '--config', type=str,
                      help='Config filename', required=True)
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    Euler.run(cfg)


if __name__ == '__main__':
    main()
