########################################################################################################################
#                                                                                                                      #
#                                       Main class for Photo linear system solver                                      #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 25.09.2021                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import torch
import scipy.sparse.linalg as linalg
from time import perf_counter
from scipy.sparse.linalg import spsolve
import copy

from .photo import photo_axisym, lambda_j_two, lambda_j_three, A_j_two, A_j_three
from .base import BasePhoto
#from ..poissonsolver.linsystem import impose_dirichlet

# Define impose dirichlet to avoid circular import problems
# TODO improve location
def impose_dirichlet(rhs: np.ndarray, bcs: dict) -> None:
    """ Impose Dirichlet boundary conditions to the rhs vector

    :param rhs: rhs vector of the Poisson equation
    :type rhs: 2D-np.ndarray
    :param bcs: dictionnary of boundary conditions
    :type bcs: dict
    """
    if 'left' in bcs:
        rhs[:, 0] = bcs['left']

    if 'right' in bcs:
        rhs[:, -1] = bcs['right']

    if 'bottom' in bcs:
        rhs[0, :] = bcs['bottom']

    if 'top' in bcs:
        rhs[-1, :] = bcs['top']


class PhotoLinSystem(BasePhoto):
    """ Class for linear system solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = self.dx * self.dy
        self.photo_model = cfg['photo_model']

        # Pressure in Torr
        self.pO2 = 150

        # Radiuses for axisymmetric computation
        self.R_nodes = copy.deepcopy(self.Y)
        self.R_nodes[0] = self.dy / 4

        # Two or three helmholtz equations depending on the photoionization model
        self.mats_photo = []
        if self.photo_model == 'two':
            self.jtot = 2
            for i in range(2):
                # Axisymmetric resolution
                self.mats_photo.append(
                    photo_axisym(self.dx, self.dy, self.nnx, self.nny, self.R_nodes,
                                    (lambda_j_two[i] * self.pO2)**2, self.scale)
                )
            self.Sphj1 = np.zeros_like(self.Sph)
            self.Sphj2 = np.zeros_like(self.Sph)

        elif self.photo_model == 'three':
            self.jtot = 3
            for i in range(3):
                # Axisymmetric resolution
                self.mats_photo.append(
                    photo_axisym(self.dx, self.dy, self.nnx, self.nny, self.R_nodes,
                                    (lambda_j_three[i] * self.pO2)**2, self.scale)
                )
            self.Sphj1 = np.zeros_like(self.Sph)
            self.Sphj2 = np.zeros_like(self.Sph)
            self.Sphj3 = np.zeros_like(self.Sph)

        # Boundary conditions imposition
        self.impose_dirichlet = impose_dirichlet

    def solve(self, ioniz_rate: np.ndarray, bcs: dict):
        """ Solve the Poisson equation with ioniz_rate as charge density / epsilon_0

        :param ioniz_rate: - rho / epsilon_0
        :type ioniz_rate: np.ndarray
        :param bcs: Dictionnary of boundary conditions
        :type bcs: dict
        """
        self.ioniz_rate = ioniz_rate
        if self.photo_model == 'two':
            for i in range(2):
                rhs = - self.ioniz_rate * A_j_two[i] * self.pO2**2 * self.scale
                self.impose_dirichlet(rhs, bcs)
                if i == 0:
                    self.Sphj1 = spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)
                elif i == 1:
                    self.Sphj2 = spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)
                self.Sph = self.Sphj1 + self.Sphj2
        elif self.photo_model == 'three':
            for i in range(3):
                rhs = - self.ioniz_rate * A_j_three[i] * self.pO2**2 * self.scale
                impose_dirichlet(rhs, bcs)
                if i == 0:
                    self.Sphj1 = spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)
                elif i == 1:
                    self.Sphj2 = spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)
                elif i == 2:
                    self.Sphj3 = spsolve(self.mats_photo[i], rhs.reshape(-1)).reshape(self.nny, self.nnx)
                self.Sph = self.Sphj1 + self.Sphj2 + self.Sphj3
        elif self.photo_model == 'custom':
            target = np.zeros_like(self.ioniz_rate[0])
            rhs = (- self.ioniz_rate[0])
            # As all the field consists on the lambda value, just select a point
            # (BC are avoided just in case)
            lambda_in = float(self.ioniz_rate[1, -5, -5])
            self.impose_dirichlet(rhs, bcs)
            mats_photo = photo_axisym(self.dx, self.dy, self.nnx, self.nny, self.R_nodes, (lambda_in)**2, self.scale)
            Sph = spsolve(mats_photo, rhs.reshape(-1)).reshape(self.nny, self.nnx)
            return Sph
        elif self.photo_model == 'custom_train':
            target = torch.zeros_like(self.ioniz_rate[:,0].unsqueeze(1))
            for batch in range(self.ioniz_rate.size(0)):
                rhs = (- self.ioniz_rate[batch, 0]).cpu().detach().numpy()
                # As all the field consists on the lambda value, just select a point
                # (BC are avoided just in case)
                lambda_in = float(self.ioniz_rate[batch, 1, -5, -5])
                self.impose_dirichlet(rhs, bcs)
                mats_photo = photo_axisym(self.dx, self.dy, self.nnx, self.nny, self.R_nodes, (lambda_in)**2, self.scale)
                Sph = spsolve(mats_photo, rhs.reshape(-1)).reshape(self.nny, self.nnx)
                target[batch] = torch.from_numpy(Sph).unsqueeze(0).cuda()
            return target