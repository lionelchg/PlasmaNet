########################################################################################################################
#                                                                                                                      #
#                                          Euler equations related routines                                            #
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
from grid import Grid
from metric import compute_voln
from plot import plot_ax_scalar, plot_ax_scalar_1D


@njit(cache=True)
def compute_timestep(cfl, dx, U, press, gamma):
    speed = np.sqrt(gamma * press / U[0]) + np.sqrt((U[1]**2 + U[2]**2) / U[0])
    dt = cfl * dx / np.max(speed[:])
    return dt

@njit(cache=True)
def compute_flux(U, gamma, r, F, press, Tgas):
    """ Compute the 2D flux of the Euler equations 
    as well as pressure and temperature """
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
def compute_res(U, F, press, dt, volc, snc, ncx, ncy, gamma, ndim, nvert, res, res_c, U_c):
    """ Compute residuals for the Lax-Wendroff scheme in a cell-vertex fashion """
    dF_c = np.zeros((4, 2))
    for i in range(ncx):
        for j in range(ncy):
            U_c[:, j, i] = 0.25 * (U[:, j, i] + U[:, j + 1, i]
                                  + U[:, j, i + 1] + U[:, j + 1, i + 1])
            res_c[:, j, i] = - (F[:, 0, j, i] * snc[0, 0] +  F[:, 1, j, i] * snc[0, 1]
                    + F[:, 0, j, i + 1] * snc[1, 0] +  F[:, 1, j, i + 1] * snc[1, 1]
                    + F[:, 0, j + 1, i + 1] * snc[2, 0] +  F[:, 1, j + 1, i + 1] * snc[2, 1]
                    + F[:, 0, j + 1, i] * snc[3, 0] +  F[:, 1, j + 1, i] * snc[3, 1])

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
            rho    = U_c[0, j, i]
            u      = U_c[1, j, i] / rho
            v      = U_c[2, j, i] / rho
            H      = U_c[3, j, i] / rho + press_c / rho
            beta   = gamma - 1
            alpha  = 0.5 * (u**2 + v**2)

            drho   = res_c[0, j, i]
            drhou  = res_c[1, j, i]
            drhov  = res_c[2, j, i]
            drhoE  = res_c[3, j, i]

            rhodu  = drhou - u*drho
            rhodv  = drhov - v*drho
            drhouu = u*drhou + u*rhodu
            drhovv = v*drhov + v*rhodv
            drhouv = u*drhov + v*rhodu

            dP     = beta * (drhoE - u*drhou -v*drhov) + beta * alpha * drho
        
            drhoH  = drhoE + dP
            drhoHu = H*rhodu + u*drhoH
            drhoHv = H*rhodv + v*drhoH

            dF_c[0, 0] = drhou
            dF_c[1, 0] = drhouu + dP
            dF_c[2, 0] = drhouv
            dF_c[3, 0] = drhoHu

            dF_c[0, 1] = drhov
            dF_c[1, 1] = drhouv
            dF_c[2, 1] = drhovv + dP
            dF_c[3, 1] = drhoHv

            res[:, j, i] -= 0.5 * ( dF_c[:, 0] * snc[0, 0] + dF_c[:, 1] * snc[0, 1] )
            res[:, j, i + 1] -= 0.5 * ( dF_c[:, 0] * snc[1, 0] + dF_c[:, 1] * snc[1, 1] )
            res[:, j + 1, i + 1] -= 0.5 * ( dF_c[:, 0] * snc[2, 0] + dF_c[:, 1] * snc[2, 1] )
            res[:, j + 1, i] -= 0.5 * ( dF_c[:, 0] * snc[3, 0] + dF_c[:, 1] * snc[3, 1] )
    

class Euler(Grid):
    # @njit(cache=True)
    def __init__(self, config):
        super().__init__(config)
        # Metric properties
        self.nvert = 4
        self.voln = compute_voln(self.X, self.dx, self.dy)
        self.volc = self.dx * self.dy
        # A bit of difference compared to avbp the nodal normal has been divided by ndim
        # it causes less calculations afterwards
        self.snc = np.array([[self.dx, self.dy], [-self.dx, self.dy], 
                    [-self.dx, -self.dy], [self.dx, -self.dy]]) / self.ndim
    
        # Boundary conditions
        self.BC = config['BC']
        if self.BC == 'full_perio':
            self.voln[:] = self.dx * self.dy
        else:
            self.voln = compute_voln(self.X, self.dx, self.dy)

        # Creation of the figures directory and numbering of outputs
        self.fig_dir = 'figures/' + config['casename']
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.number = 1
        self.save_type = config['output']['save']
        self.verbose = config['output']['verbose']
        self.period = config['output']['period']

        # Timestep calculation through cfl and fourier
        self.cfl = config['params']['cfl']
        self.dtsum = 0

        # Number of iterations
        self.nit = config['params']['nit']

        # Scalar and Residual declaration - Mixture parameters
        self.gamma = 7 / 5
        self.neqs = 4
        self.W = 0.029 # kg/mol
        self.r = co.R / self.W
        self.U = np.zeros((self.neqs, self.nny, self.nnx))
        self.U_c = np.zeros((self.neqs, self.ncy, self.ncx))
        self.F = np.zeros((self.neqs, self.ndim, self.nny, self.nnx))
        self.res = np.zeros_like(self.U)
        self.res_c = np.zeros((self.neqs, self.ncy, self.ncx))
        self.press, self.Tgas = np.zeros_like(self.X), np.zeros_like(self.X)
        self.dF_c = np.zeros((self.neqs, self.ndim))
        self.dt = 0

    def print_init(self):
        # Print header to sum up the parameters
        print(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        print(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        print(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e} -- CFL = {self.cfl:.2e}')
        print('------------------------------------')
        print('Start of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))


    # @njit(cache=True)
    def compute_flux(self):
        """ Compute the 2D flux of the Euler equations 
        as well as pressure and temperature """

        F = self.F
        U = self.U

        self.press = (self.gamma - 1) * (U[3] - (U[1]**2 + U[2]**2) / 2 / U[0])
        self.Tgas = self.press / U[0] / self.r

        # rhou - rhov
        F[0, 0] = U[1]
        F[0, 1] = U[2]
        # rho u^2 + p - rho u v
        F[1, 0] = U[1]**2 / U[0] + self.press
        F[1, 1] = U[1] * U[2] / U[0]
        # rho u^2 + p - rho u v
        F[2, 0] = U[1] * U[2] / U[0]
        F[2, 1] = U[2]**2 / U[0] + self.press
        # u(rho E + p) - v(rho E + p)
        F[3, 0] = U[1] / U[0] * (U[3] + self.press)
        F[3, 1] = U[2] / U[0] * (U[3] + self.press)

    # @njit(cache=True)
    def compute_timestep(self):
        press = self.press
        U = self.U
        speed = np.sqrt(self.gamma * press / U[0]) + np.sqrt((U[1]**2 + U[2]**2) / U[0])
        self.dt = self.cfl * self.dx / np.max(speed[:])
        self.dtsum += self.dt

    # @njit(cache=True)
    def compute_res(self):
        """ Compute residuals for the Lax-Wendroff scheme in a cell-vertex fashion """
        U = self.U
        U_c = self.U_c
        res_c = self.res_c
        res = self.res
        F = self.F
        dF_c = self.dF_c
        snc = self.snc

        for i in range(self.ncx):
            for j in range(self.ncy):
                U_c[:, j, i] = 0.25 * (U[:, j, i] + U[:, j + 1, i]
                                    + U[:, j, i + 1] + U[:, j + 1, i + 1])
                res_c[:, j, i] = - (F[:, 0, j, i] * snc[0, 0] +  F[:, 1, j, i] * snc[0, 1]
                        + F[:, 0, j, i + 1] * snc[1, 0] +  F[:, 1, j, i + 1] * snc[1, 1]
                        + F[:, 0, j + 1, i + 1] * snc[2, 0] +  F[:, 1, j + 1, i + 1] * snc[2, 1]
                        + F[:, 0, j + 1, i] * snc[3, 0] +  F[:, 1, j + 1, i] * snc[3, 1])

                # Add central term
                res[:, j, i] += res_c[:, j, i] / self.nvert
                res[:, j, i + 1] += res_c[:, j, i] / self.nvert
                res[:, j + 1, i + 1] += res_c[:, j, i] / self.nvert
                res[:, j + 1, i] += res_c[:, j, i] / self.nvert
        
        res_c *= self.dt / self.volc

        # Add second order diffusion term
        for i in range(self.ncx):
            for j in range(self.ncy):
                press_c = (self.gamma - 1) * (U_c[3, j, i] - 
                        (U[1, j, i]**2 + U[2, j, i]**2) / 2 / U[0, j, i])
                rho    = U_c[0, j, i]
                u      = U_c[1, j, i] / rho
                v      = U_c[2, j, i] / rho
                H      = U_c[3, j, i] / rho + press_c / rho
                beta   = self.gamma - 1
                alpha  = 0.5 * (u**2 + v**2)

                drho   = res_c[0, j, i]
                drhou  = res_c[1, j, i]
                drhov  = res_c[2, j, i]
                drhoE  = res_c[3, j, i]

                rhodu  = drhou - u*drho
                rhodv  = drhov - v*drho
                drhouu = u*drhou + u*rhodu
                drhovv = v*drhov + v*rhodv
                drhouv = u*drhov + v*rhodu

                dP     = beta * (drhoE - u*drhou -v*drhov) + beta * alpha * drho
            
                drhoH  = drhoE + dP
                drhoHu = H*rhodu + u*drhoH
                drhoHv = H*rhodv + v*drhoH

                dF_c[0, 0] = drhou
                dF_c[1, 0] = drhouu + dP
                dF_c[2, 0] = drhouv
                dF_c[3, 0] = drhoHu

                dF_c[0, 1] = drhov
                dF_c[1, 1] = drhouv
                dF_c[2, 1] = drhovv + dP
                dF_c[3, 1] = drhoHv

                res[:, j, i] -= 0.5 * ( dF_c[:, 0] * snc[0, 0] + dF_c[:, 1] * snc[0, 1] )
                res[:, j, i + 1] -= 0.5 * ( dF_c[:, 0] * snc[1, 0] + dF_c[:, 1] * snc[1, 1] )
                res[:, j + 1, i + 1] -= 0.5 * ( dF_c[:, 0] * snc[2, 0] + dF_c[:, 1] * snc[2, 1] )
                res[:, j + 1, i] -= 0.5 * ( dF_c[:, 0] * snc[3, 0] + dF_c[:, 1] * snc[3, 1] )

    def impose_bc_euler(self):
        """ Full periodic conditions in the 4 directions """
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
        """ Apply residual """
        self.U -= self.dt / self.voln * self.res

    def plot(self):
        """ 2D maps and 1D cuts at different y of the primitive variables """
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.U[0], r"$\rho$", geom='xy')
        press = (self.gamma - 1) * (self.U[3] - (self.U[1]**2 + self.U[2]**2) / 2 / self.U[0])
        plot_ax_scalar(fig, axes[0][1], self.X, self.Y, press, "$P$", geom='xy')
        plot_ax_scalar(fig, axes[1][0], self.X, self.Y, self.U[1] / self.U[0], "$u$", geom='xy')
        plot_ax_scalar(fig, axes[1][1], self.X, self.Y, self.U[2] / self.U[0], "$v$", geom='xy')
        plt.suptitle(rf'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(self.fig_dir + f'2D_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        plot_ax_scalar_1D(fig, axes[0][0], self.X, [0.25, 0.5], self.U[0], r"$\rho$")
        press = (self.gamma - 1) * (self.U[3] - (self.U[1]**2 + self.U[2]**2) / 2 / self.U[0])
        plot_ax_scalar_1D(fig, axes[0][1], self.X, [0.25, 0.5], press, "$P$")
        plot_ax_scalar_1D(fig, axes[1][0], self.X, [0.25, 0.5], self.U[1] / self.U[0], "$u$")
        plot_ax_scalar_1D(fig, axes[1][1], self.X, [0.25, 0.5], self.U[2] / self.U[0], "$v$")
        plt.suptitle(rf'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(self.fig_dir + f'1D_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)
        
    def postproc(self, it):
        if self.save_type == 'iteration':
            if it % self.period == 0 or it == self.nit:
                if self.verbose:
                    print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, self.dt, self.dtsum, width=14))
                self.plot()
                self.number += 1
        elif self.save_type == 'time':
            if np.abs(self.dtsum - self.number * self.period) < 0.5 * self.dt or it == self.nit:
                if self.verbose:
                    print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, self.dt, self.dtsum, width=14))
                self.plot()
                self.number += 1
        elif self.save_type == 'none':
            pass