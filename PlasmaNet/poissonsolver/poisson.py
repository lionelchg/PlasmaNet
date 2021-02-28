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

from .linsystem import matrix_cart, matrix_axisym, dc_bc
from .operators import grad, lapl
from .base import BasePoisson
from .plot import plot_modes


class Poisson(BasePoisson):
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny, config, nmax=None):
        super().__init__(xmin, xmax, nnx, ymin, ymax, nny, nmax)
        self.config = config
        self.scale = self.dx * self.dy
        if self.config == 'cart_dirichlet':
            self.mat = matrix_cart(self.dx, self.dy, nnx, nny, self.scale)
        elif self.config == 'cart_3d1n':
            self.mat = matrix_cart(self.dx, self.dy, nnx, nny, self.scale, down_bc='neumann')
        elif self.config == 'axi_dirichlet':
            self.R_nodes = copy.deepcopy(self.Y)
            self.R_nodes[0] = self.dy / 4
            self.mat = matrix_axisym(self.dx, self.dy, self.nnx, self.nny, 
                                self.R_nodes, self.scale)
        self.bc = dc_bc

    def solve(self, physical_rhs, *args):
        rhs = - physical_rhs * self.scale
        self.physical_rhs = physical_rhs.reshape(self.nny, self.nnx)
        self.bc(rhs, self.nnx, self.nny, args)
        self.potential = spsolve(self.mat, rhs).reshape(self.nny, self.nnx)

    def L2error(self, th_potential):
        return np.sqrt(np.sum(self.compute_voln() * 
                    (self.potential - th_potential)**2)) / self.Lx / self.Ly


class DatasetPoisson(Poisson):
    """ Class for dataset of poisson rhs and potentials (contains
    different plotting of modes) """
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny, config, nmax=None):
        super().__init__(xmin, xmax, nnx, ymin, ymax, nny, config, nmax)
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
