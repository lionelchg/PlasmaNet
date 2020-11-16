import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.sparse.linalg import spsolve

from poissonsolver.linsystem import matrix_cart, matrix_axisym, dc_bc
from poissonsolver.operators import grad, lapl
from poissonsolver.base import BasePoisson

class Poisson(BasePoisson):
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny, config):
        super().__init__(xmin, xmax, nnx, ymin, ymax, nny)
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
        self.bc(rhs, self.nnx, self.nny, args)
        self.potential = spsolve(self.mat, rhs).reshape(self.nny, self.nnx)
    
    def L2error(self, th_potential):
        return np.sqrt(np.sum(self.compute_voln() * 
                    (self.potential - th_potential)**2)) / self.Lx / self.Ly