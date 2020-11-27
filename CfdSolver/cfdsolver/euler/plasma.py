import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
from scipy.sparse.linalg import spsolve
from numba import njit

from .euler import Euler
from cfdsolver.scalar.init import gaussian
from poissonsolver.linsystem import matrix_cart, dc_bc
from ..base.base_plot import plot_ax_scalar, plot_ax_scalar_1D, plot_ax_vector_arrow
from ..base.operators import grad

class PlasmaEuler(Euler):
    def __init__(self, config):
        super().__init__(config)
        # Background electric field and matrix construction
        zeros_x = np.zeros_like(self.x)
        zeros_y = np.zeros_like(self.y)
        self.down, self.up = zeros_x, zeros_x
        self.left, self.right = zeros_y, zeros_y
        self.scale = self.dx * self.dy
        self.mat_poisson = matrix_cart(self.dx, self.dy, 
                            self.nnx, self.nny, self.scale)
        self.bc = dc_bc

        self.m_e = co.m_e
        self.W = self.m_e * co.N_A
        self.n_back = config['params']['n_back']
        self.n_pert = config['params']['n_pert']
        x0, y0, sigma_x, sigma_y = 5e-3, 5e-3, 1e-3, 1e-3
        n_electron = gaussian(self.X, self.Y, self.n_pert, x0, 
                            y0, sigma_x, sigma_y) + self.n_back
        self.U[0] = self.m_e * n_electron

        self.omega_p = np.sqrt(self.n_back * co.e**2 / self.m_e / co.epsilon_0)
        self.dt = 2 * np.pi / self.omega_p / config['params']['nt_oscill']

        self.time = np.zeros(self.nit)
        # first is center n_e, then n_e at offsets and normE at offsets
        self.temporals = np.zeros((self.nit, 7))
        self.nnx_mid, self.nny_mid = int(self.nnx / 2), int(self.nny / 2)
        self.offsets = [0.2, 0.4, 0.6]

    def print_init(self):
        """ Print header to sum up the parameters. """
        print(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        print(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        print(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e}')
        print(f'dt = {self.dt:.2e} s - T_p = {2 * np.pi / self.omega_p:.2e} s - omega_p = {self.omega_p:.2e} rad.s-1')
        print('------------------------------------')
        print('Start of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))


    def solve_poisson(self):
        """ Solve the Poisson equation in axisymmetric configuration. """
        self.physical_rhs = - (self.U[0] / self.m_e - self.n_back).reshape(-1) * co.e / co.epsilon_0
        self.rhs = - self.physical_rhs * self.scale
        self.bc(self.rhs, self.nnx, self.nny, self.down, self.up, self.left, self.right)
        self.potential = spsolve(self.mat_poisson, self.rhs).reshape(self.nny, self.nnx)
        self.E_field = - grad(self.potential, self.dx, self.dy, self.nnx, self.nny)
        self.physical_rhs = self.physical_rhs.reshape((self.nny, self.nnx))
        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1: self.E_max = np.max(self.E_norm)

    def compute_flux_cold(self):
        """ Compute the 2D flux of the Euler equations but without pressure """
        F = self.F
        U = self.U
        compute_flux_cold(U, self.gamma, self.r, F)

    def compute_EM_source(self):
        """ Compute electro-magnetic source terms in vertex-centered approximation """
        self.res[1] += self.U[0] / self.m_e * co.e * self.E_field[0] * self.voln
        self.res[2] += self.U[0] / self.m_e * co.e * self.E_field[1] * self.voln
        self.res[3] += co.e * (self.U[1] * self.E_field[0] + self.U[2] * self.E_field[1]) / self.m_e * self.voln
    
    def plot(self):
        """ 2D maps and 1D cuts at different y of the primitive variables. """
        n_e = self.U[0] / self.m_e - self.n_back
        E = self.E_field
        E_norm = self.E_norm
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, n_e, r"$n_e$", geom='xy', max_value=1.1*self.n_pert)
        plot_ax_scalar_1D(fig, axes[0][1], self.X, [0.5], n_e, r"$n_e$", ylim=[-1.1*self.n_pert, 1.1*self.n_pert])
        plot_ax_vector_arrow(fig, axes[1][0], self.X, self.Y, E, 'Electric field', max_value=1.1*self.E_max)
        plot_ax_scalar_1D(fig, axes[1][1], self.X, [0.5], E_norm, r"$|\mathbf{E}|$", ylim=[0, 1.1*self.E_max])
        plt.suptitle(rf'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(self.fig_dir + f'variables_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def mean_temp(var, nny_mid, nnx_mid, offset):
        return 0.25 * (var[nny_mid - offset, nnx_mid - offset] 
                        + var[nny_mid - offset, nnx_mid + offset] 
                        + var[nny_mid + offset, nnx_mid - offset] 
                        + var[nny_mid + offset, nnx_mid + offset])

    def temporal_variables(self, it):
        nep = self.U[0, :, :] / self.m_e - self.n_back
        self.temporals[it - 1, 0] = nep[self.nny_mid, self.nnx_mid]
        for i, offset in enumerate(self.offsets):
            ioffset = int(offset * (self.nnx - 1) / 2)
            self.temporals[it - 1, 1 + i] = self.mean_temp(nep, self.nny_mid, 
                                                        self.nnx_mid, ioffset)
            self.temporals[it - 1, 4 + i] = self.mean_temp(self.E_norm, 
                                        self.nny_mid, self.nnx_mid, ioffset)
    
    @staticmethod
    def ax_prop(ax, xlabel, ylabel, axtitle):
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(axtitle)
        ax.legend()

    def plot_temporal(self):
        fig, axes = plt.subplots(ncols=2, figsize=(12, 7))
        axes[0].plot(self.time, self.temporals[:, 0], label='r = 0')
        for i in range(3):
            offset = self.offsets[i]
            axes[0].plot(self.time, self.temporals[:, 1 + i], label=f'r = {offset:.1f} rmax')
            axes[1].plot(self.time, self.temporals[:, 4 + i], label=f'r = {offset:.1f} rmax')
        self.ax_prop(axes[0], '$t$ [s]', r'$n_e$ [m$^{-3}$]', r'Temporal evolution of $n_e$')
        self.ax_prop(axes[1], '$t$ [s]', r'$|\mathbf{E}|$ [V.m$^{-1}$]', 
                                        r'Temporal evolution of $|\mathbf{E}|$')
        fig.savefig(self.fig_dir + 'temporals', bbox_inches='tight')

@njit(cache=True)
def compute_flux_cold(U, gamma, r, F):
    """ Compute the 2D flux of the Euler equations 
    as well as pressure and temperature. """
    # rhou - rhov
    F[0, 0] = U[1]
    F[0, 1] = U[2]
    # rho u^2 + p - rho u v
    F[1, 0] = U[1]**2 / U[0]
    F[1, 1] = U[1] * U[2] / U[0]
    # rho u^2 + p - rho u v
    F[2, 0] = U[1] * U[2] / U[0]
    F[2, 1] = U[2]**2 / U[0]
    # u(rho E + p) - v(rho E + p)
    F[3, 0] = U[1] / U[0] * U[3]
    F[3, 1] = U[2] / U[0] * U[3]