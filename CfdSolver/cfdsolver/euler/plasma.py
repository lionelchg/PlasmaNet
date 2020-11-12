import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
from scipy.sparse.linalg import spsolve

from .euler import Euler
from cfdsolver.scalar.init import gaussian
from poissonsolver.linsystem import matrix_cart, dc_bc
from ..base.base_plot import plot_ax_scalar, plot_ax_scalar_1D
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
        self.n_back = config['params']['n_back']
        self.n_pert = config['params']['n_pert']
        x0, y0, sigma_x, sigma_y = 5e-3, 5e-3, 1e-3, 1e-3
        n_electron = gaussian(self.X, self.Y, self.n_pert, x0, 
                            y0, sigma_x, sigma_y) + self.n_back
        self.U[0] = self.m_e * n_electron

        self.omega_p = np.sqrt(self.n_back * co.e**2 / self.m_e / co.epsilon_0)
        self.dt = 2 * np.pi / self.omega_p / 100

        self.time = np.zeros(self.nit)
        self.n_center = np.zeros(self.nit)
        self.E_center = np.zeros(self.nit)
        self.nnx_mid, self.nny_mid = int(self.nnx / 2), int(self.nny / 2)

    def print_init(self):
        """ Print header to sum up the parameters. """
        print(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        print(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        print(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e}')
        print(f'dt = {self.dt:.2e} s - omega_p = {self.omega_p:.2e} rad.s-1')
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

    def compute_EM_source(self):
        """ Compute electro-magnetic source terms in vertex-centered approximation """
        self.res[1] += self.U[0] / self.m_e * co.e * self.E_field[0] * self.voln
        self.res[2] += self.U[0] / self.m_e * co.e * self.E_field[1] * self.voln
        self.res[3] += co.e * (self.U[1] * self.E_field[0] + self.U[2] * self.E_field[1]) / self.m_e * self.voln
    
    def plot(self):
        """ 2D maps and 1D cuts at different y of the primitive variables. """
        n_e = self.U[0] / self.m_e - self.n_back
        E_norm = self.E_norm
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, n_e, r"$n_e$", geom='xy', max_value=self.n_pert)
        plot_ax_scalar_1D(fig, axes[0][1], self.X, [0.5], n_e, r"$n_e$")
        plot_ax_scalar(fig, axes[1][0], self.X, self.Y, E_norm, r"$|\mathbf{E}|$", geom='xy')
        plot_ax_scalar_1D(fig, axes[1][1], self.X, [0.5], E_norm, r"$|\mathbf{E}|$")
        plt.suptitle(rf'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(self.fig_dir + f'variables_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)
    
    def center_variables(self, it):
        self.n_center[it - 1] = self.U[0, self.nny_mid, self.nnx_mid] / self.m_e
        self.E_center[it - 1] = self.E_norm[self.nny_mid, self.nnx_mid]
    
    def plot_temporal(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.time, self.n_center, label='n_e', color='blue')

        ax1 = ax.twinx()
        ax1.plot(self.time, self.E_center, label=r'|\mathbf{E}|', color='orange')

        fig.savefig(self.fig_dir + 'center', bbox_inches='tight')