########################################################################################################################
#                                                                                                                      #
#                                            Main class for Poisson solver                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.sparse.linalg import spsolve

from ..common.plot import plot_modes
from .linsystem import matrix_cart, matrix_axisym, impose_dc_bc
from .base import BasePoisson


class PoissonLinSystem(BasePoisson):
    """ Class for linear system solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = self.dx * self.dy
        if cfg['mat'] == 'cart_dirichlet':
            self.mat = matrix_cart(self.dx, self.dy, self.nnx, self.nny, self.scale)
        elif cfg['mat'] == 'cart_3d1n':
            self.mat = matrix_cart(self.dx, self.dy, self.nnx, self.nny, self.scale, down_bc='neumann')
        elif cfg['mat'] == 'axi_dirichlet':
            self.R_nodes = copy.deepcopy(self.Y)
            self.R_nodes[0] = self.dy / 4
            self.mat = matrix_axisym(self.dx, self.dy, self.nnx, self.nny, 
                                self.R_nodes, self.scale)
        self.dirichlet_bc = impose_dc_bc

    def solve(self, physical_rhs, *args):
        """ Solve the Poisson problem with physical_rhs and args
        boundary conditions (up to 4 boundary conditions for each side)

        :param physical_rhs: - rho / epsilon_0 
        :type physical_rhs: ndarray
        """
        assert len(args) <= 4
        rhs = - physical_rhs * self.scale
        self.physical_rhs = physical_rhs.reshape(self.nny, self.nnx)
        self.dirichlet_bc(rhs, self.nnx, self.nny, args)
        self.potential = spsolve(self.mat, rhs).reshape(self.nny, self.nnx)


class DatasetPoisson(PoissonLinSystem):
    """ Class for dataset of poisson rhs and potentials (contains
    different plotting of modes) """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Mean, min and max
        self.coeffs_rhs_dset = np.zeros((2, self.nmax, self.nmax))
        self.coeffs_pot_dset = np.zeros((2, self.nmax, self.nmax))
        self.nfourier_comput = 0

    def compute_modes(self):
        """ Compute the fourier coefficients of rhs and potential """
        super().compute_modes()
        self.nfourier_comput += 1
        self.coeffs_rhs_dset[0] += self.coeffs_rhs
        self.coeffs_rhs_dset[1] = np.maximum(self.coeffs_rhs_dset[1], self.coeffs_rhs)
        self.coeffs_pot_dset[0] += self.coeffs_pot
        self.coeffs_pot_dset[1] = np.maximum(self.coeffs_pot_dset[1], self.coeffs_pot)

    def plot_pmodes(self, figname):
        """ Plot the potential and rhs modes from 2D
        Fourier expansion """
        self.coeffs_rhs_dset[0] /= self.nfourier_comput
        self.coeffs_pot_dset[0] /= self.nfourier_comput
        fig = plt.figure(figsize=(12, 14))
        ax1 = fig.add_subplot(221, projection='3d')
        plot_modes(ax1, self.N, self.M, self.coeffs_rhs_dset[0], "RHS modes")
        ax2 = fig.add_subplot(222, projection='3d')
        plot_modes(ax2, self.N, self.M, self.coeffs_pot_dset[0], "Potential modes")
        ax3 = fig.add_subplot(223, projection='3d')
        plot_modes(ax3, self.N, self.M, self.coeffs_rhs_dset[1], "RHS modes", 'Purples')
        ax4 = fig.add_subplot(224, projection='3d')
        plot_modes(ax4, self.N, self.M, self.coeffs_pot_dset[1], "Potential modes", 'Purples')

        fig.tight_layout()
        fig.savefig(figname, bbox_inches='tight')
        plt.close()

    def sum_series(self, coefs):
        """ Series of rhs from Fourier resolution given coeffs """
        series = np.zeros_like(self.X)
        for n in range(1, self.nmax + 1):
            for m in range(1, self.nmax + 1):
                series += coefs[n - 1, m - 1] * np.sin(n * np.pi * self.X / self.Lx) * np.sin(m * np.pi * self.Y / self.Ly)
        return series

    def pot_series(self, coefs):
        """ Series of potential from Fourier resolution given coeffs """
        series = np.zeros_like(self.X)
        for n in range(1, self.nmax + 1):
            for m in range(1, self.nmax + 1):
                series += (coefs[n - 1, m - 1] * np.sin(n * np.pi * self.X / self.Lx) 
                    * np.sin(m * np.pi * self.Y / self.Ly) / ((n * np.pi / self.Lx)**2 + (m * np.pi / self.Ly)**2))
        return series
