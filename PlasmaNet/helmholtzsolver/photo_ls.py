########################################################################################################################
#                                                                                                                      #
#                                       Main class for Photo linear system solver                                      #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 25.09.2021                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import scipy.sparse.linalg as linalg
from time import perf_counter
from scipy.sparse.linalg import spsolve
import copy

from .photo import photo_axisym, lambda_j_two, lambda_j_three, A_j_two, A_j_three
from .base import BasePhoto
from ..poissonsolver.linsystem import impose_dirichlet

class PhotoLinSystem(BasePhoto):
    """ Class for linear system solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = self.dx * self.dy
        self.photo_model = cfg['model']

        # Pressure in Torr
        self.pO2 = 150

        # Radiuses for axisymmetric computation
        self.R_nodes = copy.deepcopy(self.Y)
        self.R_nodes[0] = self.dy / 4

        # Two or three helmholtz equations depending on the photoionization model
        self.mats_photo = []
        if self.photo_model == 'two':
            for i in range(2):
                # Axisymmetric resolution
                self.mats_photo.append(
                    photo_axisym(self.dx, self.dy, self.nnx, self.nny, self.R_nodes,
                                    (lambda_j_two[i] * self.pO2)**2, self.scale)
                )
        elif self.photo_model == 'three':
            for i in range(3):
                # Axisymmetric resolution
                self.mats_photo.append(
                    photo_axisym(self.dx, self.dy, self.nnx, self.nny, self.R_nodes,
                                    (lambda_j_three[i] * self.pO2)**2, self.scale)
                )
        # Boundary conditions imposition
        self.impose_dirichlet = impose_dirichlet

    def solve(self, ioniz_rate: np.ndarray, bcs: dict):
        """ Solve the Poisson equation with ioniz_rate as charge density / epsilon_0

        :param ioniz_rate: - rho / epsilon_0
        :type ioniz_rate: np.ndarray
        :param bcs: Dictionnary of boundary conditions
        :type bcs: dict
        """
        self.Sph[:] = 0
        self.ioniz_rate = ioniz_rate
        if self.photo_model == 'two':
            for i in range(2):
                rhs = - self.ioniz_rate * A_j_two[i] * self.pO2**2 * self.scale
                self.impose_dirichlet(rhs, bcs)
                self.Sph += spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)

        elif self.photo_model == 'three':
            for i in range(3):
                rhs = - self.ioniz_rate * A_j_three[i] * self.pO2**2 * self.scale
                impose_dirichlet(rhs, bcs)
                self.Sph += spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)
