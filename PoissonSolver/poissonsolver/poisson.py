import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse.linalg import spsolve

from poissonsolver.linsystem import matrix_cart, matrix_axisym, dc_bc
from poissonsolver.operators import grad, lapl
from poissonsolver.base import BasePoisson, fourier_coef_2D
from poissonsolver.plot import plot_modes

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

    def compute_modes(self):
        """ Compute the fourier coefficients of rhs and potential """
        if self.nmax is not None:
            for i in self.nrange:
                for j in self.nrange:
                    self.coeffs_rhs[j - 1, i - 1] += fourier_coef_2D(self.X, self.Y, self.Lx, self.Ly, self.voln, self.physical_rhs, i, j)
                    self.coeffs_pot[j - 1, i - 1] += self.coeffs_rhs[j - 1, i - 1] / np.pi**2 / ((i / self.Lx)**2 + (j / self.Ly)**2)
        else:
            print("Class is not initialized for computing modes")
    

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

