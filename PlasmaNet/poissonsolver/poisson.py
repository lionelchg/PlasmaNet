########################################################################################################################
#                                                                                                                      #
#                                            Main class for Poisson solver                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.sparse.linalg as linalg
from time import perf_counter

from .base import BasePoisson
from .linsystem import (cartesian_matrix, matrix_axisym, impose_dirichlet,
                        matrix_cart_perio, matrix_cart_perio_x)
from ..common.plot import plot_modes
from ..common.utils import create_dir


class PoissonLinSystem(BasePoisson):
    """ Class for linear system solver of Poisson problem

    :param BasePoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = self.dx * self.dy
        self.perio = 'perio' in cfg['bcs']
        self.bcs = cfg['bcs']

        # Matrix construction
        self.geom = cfg['geom']
        if cfg['geom'] in ['cartesian', 'xy'] and not self.perio:
            # Reformat boundary conditions if similar on all boundaries
            if isinstance(cfg['bcs'], str):
                bc = cfg['bcs']
                cfg['bcs'] = {'left': bc, 'right': bc, 'bottom': bc, 'top': bc}

            self.mat = cartesian_matrix(self.dx, self.dy, self.nnx, self.nny, self.scale, cfg['bcs'])
        elif cfg['geom'] in ['cylindrical', 'xr'] and not self.perio:
            self.R_nodes = copy.deepcopy(self.Y)
            self.R_nodes[0] = self.dy / 4
            self.mat = matrix_axisym(self.dx, self.dy, self.nnx, self.nny,
                                     self.R_nodes, self.scale)
        elif cfg['bcs'] == 'perio':
            self.mat = matrix_cart_perio(self.dx, self.dy, self.nnx - 1, self.nny - 1, self.scale)
        elif cfg['bcs'] == 'perio_x':
            self.mat = matrix_cart_perio_x(self.dx, self.dy, self.nnx - 1, self.nny, self.scale)

        # Boundary conditions imposition
        self.impose_dirichlet = impose_dirichlet

        # Solver configuration
        if "solver_type" in cfg:
            if cfg["solver_type"] == "direct":
                # Initializes direct solver
                self.solver_type = "direct"
                # pass the useUmfpack and assumeSortedIndices options
                linalg.use_solver(**cfg["solver_options"])
            else:
                self.solver_type = cfg["solver_type"]
                self.solver_options = cfg["solver_options"]  # Options to pass to the solver
        else:
            self.solver_type = "direct"

    def solve(self, physical_rhs: np.ndarray, bcs: dict):
        """ Solve the Poisson equation with physical_rhs as charge density / epsilon_0

        :param physical_rhs: - rho / epsilon_0
        :type physical_rhs: np.ndarray
        :param bcs: Dictionnary of boundary conditions
        :type bcs: dict
        """
        if not self.perio:
            self.physical_rhs = physical_rhs
            rhs = - physical_rhs * self.scale
            self.impose_dirichlet(rhs, bcs)
            self.potential = self._solve(self.mat, rhs.reshape(-1)).reshape(self.nny, self.nnx)
        elif self.bcs == 'perio':
            self.physical_rhs = physical_rhs
            rhs = - physical_rhs[:-1, :-1] * self.scale
            self.potential[:-1, :-1] = self._solve(self.mat, rhs.reshape(-1)).reshape(self.nny - 1, self.nnx - 1)
            self.potential[:-1, -1] = self.potential[:-1, 0]
            self.potential[-1, :] = self.potential[0, :]
        elif self.bcs == 'perio_x':
            rhs = - physical_rhs[:, :-1] * self.scale
            self.potential[:, :-1] = self._solve(self.mat, rhs.reshape(-1)).reshape(self.nny, self.nnx - 1)
            self.potential[:, -1] = self.potential[:, 0]

    def _solve(self, A, b):
        """ Calls the required solver.

        :param cfg: Poisson configuration dict
        """
        if self.benchmark:
            solve_timer = perf_counter()

        if self.solver_type == "direct":
            x = linalg.spsolve(A, b)

        elif self.solver_type == "gmres":
            x = linalg.gmres(A, b, **self.solver_options)

        elif self.solver_type == "cg":
            x = linalg.cg(A, b, **self.solver_options)

        else:
            raise ValueError(f"Unknown solver_type {self.solver_type}")

        if self.benchmark:
            solve_timer = perf_counter() - solve_timer
            print(f"solve_time={solve_timer}")

        return x

    def run_case(self, case_dir: str, physical_rhs: np.ndarray,
                 pot_bcs: dict, plot: bool, save=True, axis='off'):
        """ Run a Poisson linear system case

        :param case_dir: Case directory
        :type case_dir: str
        :param physical_rhs: physical rhs
        :type physical_rhs: np.ndarray
        :param pot_bcs: Dirichlet boundary conditions
        :type pot_bcs: dict
        :param plot: logical for plotting
        :type plot: bool
        :param save: logical for saving rhs and pot
        :type save: bool
        """
        if self.geom == 'cylindrical':
            geom = 'xr'
        else:
            geom = 'xy'

        create_dir(case_dir)
        self.solve(physical_rhs, pot_bcs)
        if save:
            self.save(case_dir)
        if plot:
            fig_dir = case_dir + '/figures/'
            create_dir(fig_dir)
            self.plot_2D(fig_dir + '2D', geom=geom, axis=axis)
            self.plot_1D2D(fig_dir + 'full', geom=geom)


class DatasetPoisson(PoissonLinSystem):
    """ Class for dataset of poisson rhs and potentials (contains
    different plotting of modes)
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Mean, min and max
        self.coeffs_rhs_dset = np.zeros((2, len(self.mrange), len(self.nrange)))
        self.coeffs_pot_dset = np.zeros((2, len(self.mrange), len(self.nrange)))
        self.nfourier_comput = 0

    def compute_modes(self):
        """ Compute the fourier coefficients of rhs and potential """
        super().compute_modes()
        self.nfourier_comput += 1
        self.coeffs_rhs_dset[0] += self.coeffs_rhs
        self.coeffs_rhs_dset[1] = np.maximum(self.coeffs_rhs_dset[1], self.coeffs_rhs)
        self.coeffs_pot_dset[0] += self.coeffs_pot
        self.coeffs_pot_dset[1] = np.maximum(self.coeffs_pot_dset[1], self.coeffs_pot)

    def sum_series(self, coefs):
        """ Series of rhs from Fourier resolution given coefficients """
        series = np.zeros_like(self.X)
        for n in range(self.nmin, self.nmax + 1):
            for m in range(self.mmin, self.nmax + 1):
                series += coefs[m - self.mmin, n - self.nmin] * np.sin(n * np.pi * self.X / self.Lx) \
                          * np.sin(m * np.pi * self.Y / self.Ly)
        return series

    def pot_series(self, coefs):
        """ Series of potential from Fourier resolution given coeffs """
        series = np.zeros_like(self.X)
        for n in range(self.nmin, self.nmax + 1):
            for m in range(self.mmin, self.nmax + 1):
                series += (coefs[m - self.mmin, n - self.nmin] * np.sin(n * np.pi * self.X / self.Lx)
                           * np.sin(m * np.pi * self.Y / self.Ly) / ((n * np.pi / self.Lx)**2
                           + (m * np.pi / self.Ly)**2))
        return series

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
