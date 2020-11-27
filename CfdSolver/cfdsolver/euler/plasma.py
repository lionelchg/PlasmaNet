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
from ..utils import create_dir

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
        self.T_p = 2 * np.pi / self.omega_p
        self.nt_oscill = config['params']['nt_oscill']
        self.dt = 2 * np.pi / self.omega_p / self.nt_oscill

        if 'n_periods' in config['params']:
            self.n_periods = config['params']['n_periods']
            self.nit = self.n_periods * self.nt_oscill
            self.end_time = self.n_periods * self.T_p
        elif 'end_time' in config['params']:
            self.nit = int(self.end_time / self.dt)
            self.n_periods = self.end_time / self.T_p
        else:
            self.end_time = self.nit * self.dt
            self.n_periods = self.end_time / self.T_p

        self.time = np.zeros(self.nit)
        # first is center n_e, then n_e at offsets and normE at offsets
        self.temporals = np.zeros((self.nit, 7))
        self.nnx_mid, self.nny_mid = int(self.nnx / 2), int(self.nny / 2)
        self.offsets = [0.2, 0.4, 0.6]

        # datasets for deep-learning
        self.dl_save = config['output']['dl_save'] == 'yes'
        if self.dl_save:
            self.dl_dir = config['casename'] + 'dl_data/'
            create_dir(self.dl_dir)
            self.potential_list = np.zeros((self.nit, self.nny, self.nnx))
            self.physical_rhs_list = np.zeros((self.nit, self.nny, self.nnx))

    def print_init(self):
        """ Print header to sum up the parameters. """
        print(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        print(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        print(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e}')
        print(f'dt = {self.dt:.2e} s - T_p = {self.T_p:.2e} s - omega_p = {self.omega_p:.2e} rad.s-1')
        print('------------------------------------')
        print('Start of simulation')
        print('------------------------------------')
        print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))
    
    def write_init(self):
        """ Print header to sum up the parameters. """
        fp = open(self.case_dir + 'case.log', 'w')
        fp.write(f'- Number of nodes: \nnnx = {self.nnx:d}\nnny = {self.nny:d}\n')
        fp.write(f'- Bounding box: \n({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})\n')
        fp.write(f'- Mesh: \ndx = {self.dx:.2e}\ndy = {self.dy:.2e}\n')
        fp.write(f'- Plasma Oscillation: \nn_back = {self.n_back:.2e}\nn_pert = {self.n_pert:.2e}\n')
        fp.write(f'T_p = {self.T_p:.2e} s\nomega_p = {self.omega_p:.2e} rad.s-1\n')
        fp.write(f'dt = {self.dt:.2e} s\nnt_oscill = {self.nt_oscill:d}\n')
        fp.write(f'nit = {self.nit:d}\nn_periods = {self.n_periods:.1f}')
        fp.close()


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
    
    def postproc(self, it):
        super().postproc(it)
        if self.dl_save:
            self.potential_list[it - 1, :, :] = self.potential
            self.physical_rhs_list[it - 1, :, :] = self.physical_rhs

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
    def save(self):
        pass

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
        for i in range(1):
            offset = self.offsets[i]
            axes[0].plot(self.time, self.temporals[:, 1 + i], label=f'r = {offset:.1f} rmax')
        for i in range(3):
            offset = self.offsets[i]    
            axes[1].plot(self.time, self.temporals[:, 4 + i], label=f'r = {offset:.1f} rmax')
        self.ax_prop(axes[0], '$t$ [s]', r'$n_e$ [m$^{-3}$]', r'Temporal evolution of $n_e$')
        self.ax_prop(axes[1], '$t$ [s]', r'$|\mathbf{E}|$ [V.m$^{-1}$]', 
                                        r'Temporal evolution of $|\mathbf{E}|$')
        fig.savefig(self.fig_dir + 'temporals', bbox_inches='tight')
    
    def post_temporal(self):
        self.plot_temporal()
        if self.dl_save:
            np.save(self.dl_dir + 'potential.npy', self.potential_list)
            np.save(self.dl_dir + 'physical_rhs.npy', self.physical_rhs_list)

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