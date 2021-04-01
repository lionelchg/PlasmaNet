########################################################################################################################
#                                                                                                                      #
#                                              Metric related functions                                                #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import copy


class Mesh:
    def __init__(self, config):
        # Mesh properties
        self.ndim = 2
        self.nnx = config['mesh']['nnx']
        self.xmin, self.xmax = config['mesh']['xmin'], config['mesh']['xmax']
        # if there is no y properties a square is assumed with same properties on
        # x and y axis
        if 'nny' in config['mesh']: 
            self.nny = config['mesh']['nny']
            self.ymin, self.ymax = config['mesh']['ymin'], config['mesh']['ymax']
        else:
            self.nny = self.nnx
            self.ymin, self.ymax = self.xmin, self.xmax

        self.ncx, self.ncy = self.nnx - 1, self.nny - 1  # Number of cells
        self.Lx, self.Ly = self.xmax - self.xmin, self.ymax - self.ymin
        self.dx = (self.xmax - self.xmin) / self.ncx
        self.dy = (self.ymax - self.ymin) / self.ncy
        self.x = np.linspace(self.xmin, self.xmax, self.nnx)
        self.y = np.linspace(self.ymin, self.ymax, self.nny)
        # Grid construction
        self.X, self.Y = np.meshgrid(self.x, self.y)


class Grid(Mesh):
    def __init__(self, config):
        super().__init__(config)
    
        # Boundary conditions
        self.BC = config['BC']
        if self.BC == 'full_perio':
            self.voln = np.zeros_like(self.X)
            self.voln[:] = self.dx * self.dy
        else:
            self.voln = self.compute_voln(self.X, self.dx, self.dy)
        
        # Variables related to cell vertex formulation
        self.nvert = 4
        self.volc = self.dx * self.dy
        # A bit of difference compared to avbp the nodal normal has been divided by ndim
        # it causes less calculations afterwards
        self.snc = np.array([[self.dx, self.dy], [-self.dx, self.dy], 
                    [-self.dx, -self.dy], [self.dx, -self.dy]]) / self.ndim

        # Variables related to vertex centered formulation
        self.geom = config['params']['geom']
        if self.geom == 'xr':
            self.R_nodes = copy.deepcopy(self.Y)
            self.R_nodes[0] = self.dy / 4
            self.voln *= self.R_nodes
        else:
            self.R_nodes = None

        self.sij = np.array([self.dy / 2, self.dx / 2])

    @staticmethod
    def compute_voln(X, dx, dy):
        """ Computes the nodal volume associated to each node (i, j) """
        voln = np.ones_like(X) * dx * dy
        voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
            dx * dy / 2, dx * dy / 2, dx * dy / 2, dx * dy / 2
        voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
            dx * dy / 4, dx * dy / 4, dx * dy / 4, dx * dy / 4
        return voln

    def domain_ave(self, var):
        """ Returns the domain average of var """
        return np.sum(var * self.voln) / self.Lx / self.Ly
