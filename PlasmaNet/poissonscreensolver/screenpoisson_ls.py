########################################################################################################################
#                                                                                                                      #
#                            Main class for Screened Poisson solver using linear system solver                         #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 25.09.2021                                           #
#                                                                                                                      #
########################################################################################################################

import copy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.sparse.linalg as linalg
from time import perf_counter
from pathlib import Path

from ..poissonsolver.base import BasePoisson
from .linsystem import cart_matrix, axisym_matrix
from ..poissonsolver.linsystem import impose_dirichlet

from ..common.operators_numpy import grad, lapl

class ScreenPoissonLinSystem(BasePoisson):
    """ Class for linear system solver of Poisson problem

    :param BaseScreenPoisson: Base class for Poisson routines
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scale = self.dx * self.dy
        self.perio = 'perio' in cfg['bcs']
        self.bcs = cfg['bcs']
        self.coeff = cfg['coeff']

        # Matrix construction
        self.geom = cfg['geom']
        if cfg['geom'] in ['cartesian', 'xy'] and not self.perio:
            # Reformat boundary conditions if similar on all boundaries
            if isinstance(cfg['bcs'], str):
                bc = cfg['bcs']
                cfg['bcs'] = {'left': bc, 'right': bc, 'bottom': bc, 'top': bc}

            self.mat = cart_matrix(self.dx, self.dy, self.nnx, self.nny, self.coeff**2, self.scale, cfg['bcs'])
        elif cfg['geom'] in ['cylindrical', 'xr'] and not self.perio:
            self.R_nodes = copy.deepcopy(self.Y)
            self.R_nodes[0] = self.dy / 4
            self.mat = axisym_matrix(self.dx, self.dy, self.nnx, self.nny,
                                     self.R_nodes, self.coeff, self.scale)

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

    @property
    def lapl(self):
        """ Laplacian computation of the potential minus the screening term
        here for plotting

        :return: the Laplacian of the potential
        :rtype: ndarray
        """
        return (lapl(self.potential, self.dx, self.dy, self.nnx, self.nny, r=self.R_nodes)
                    - self.coeff**2 * self.potential)


    def solve(self, physical_rhs: np.ndarray, bcs: dict):
        """ Solve the Poisson equation with physical_rhs as charge density / epsilon_0

        :param physical_rhs: - rho / epsilon_0
        :type physical_rhs: np.ndarray
        :param bcs: Dictionnary of boundary conditions
        :type bcs: dict
        """
        self.physical_rhs = physical_rhs
        rhs = - physical_rhs * self.scale
        self.impose_dirichlet(rhs, bcs)
        self.potential = self._solve(self.mat, rhs.reshape(-1)).reshape(self.nny, self.nnx)

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

    def run_case(self, case_dir: Path, physical_rhs: np.ndarray,
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

        case_dir.mkdir(parents=True, exist_ok=True)
        self.solve(physical_rhs, pot_bcs)
        if save:
            self.save(case_dir)
        if plot:
            fig_dir = case_dir / 'figures'
            fig_dir.mkdir(parents=True, exist_ok=True)
            self.plot_2D(fig_dir / '2D', geom=geom, axis=axis)
            self.plot_1D2D(fig_dir / 'full', geom=geom)
