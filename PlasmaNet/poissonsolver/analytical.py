########################################################################################################################
#                                                                                                                      #
#                                              Analytical Poisson solver                                               #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from numba import njit

from poissonsolver.base import BasePoisson, fourier_coef_2D, fourier_coef_1D


class PoissonAnalytical(BasePoisson):
    """ Analytical solution of the 2D Poisson problem 
    with 4 dirichlet boundaries in cartesian rectangular geometry """
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny, 
                nmax_rhs, mmax_rhs, nmax_dirichlet, nmax=None):
        super().__init__(xmin, xmax, nnx, ymin, ymax, nny, nmax)

        self.bcs = [np.zeros(nnx), np.zeros(nnx), np.zeros(nny), np.zeros(nny)]

        # Decomposition of the potentials (rhs, down, up, left, right)
        self.potentials = np.zeros((5, nny, nnx))

        self.nmax_dirichlet = nmax_dirichlet
        self.nmax_rhs = nmax_rhs
        self.mmax_rhs = mmax_rhs
    
    def rhs_solution(self):
        """ Solve the rhs problem """
        for n in range(1, self.nmax_rhs + 1):
            for m in range(1, self.mmax_rhs + 1):
               self.potentials[0] += series_term(self.X, self.Y, self.Lx, self.Ly, self.voln, self.physical_rhs, n, m)
        self.potentials[0] /= np.pi**2
    
    def bc_solution(self):
        """ Solve the 4 BC problems """
        for n in range(1, self.nmax_dirichlet + 1):
            self.potentials[1] += series_term_ddown(self.bcs[0], self.X, self.Y, self.Lx, self.Ly, n)
            self.potentials[2] += series_term_dup(self.bcs[1], self.X, self.Y, self.Lx, self.Ly, n)
            self.potentials[3] += series_term_dleft(self.bcs[2], self.X, self.Y, self.Lx, self.Ly, n)
            self.potentials[4] += series_term_dright(self.bcs[3], self.X, self.Y, self.Lx, self.Ly, n)
        self.potentials[1:3] *= 2 / self.Lx
        self.potentials[3:] *= 2 / self.Ly

    def compute_sol(self, *args):
        """ Solves the analytical solution of the 2D dirichlet Poisson problem
        with either only rhs (len(args) == 1) only dirichlet bcs (len(args) == 4)
        or both (len(args) == 5) """
        self.potentials[:] = 0
        if len(args) == 1:
            self.physical_rhs = args[0]
            self.rhs_solution()
        elif len(args) == 4:
            self.bcs[0] = args[0]
            self.bcs[1] = args[1]
            self.bcs[2] = args[2]
            self.bcs[3] = args[3]
            self.bc_solution()
        elif len(args) == 5:
            self.physical_rhs = args[0]
            self.bcs[0] = args[1]
            self.bcs[1] = args[2]
            self.bcs[2] = args[3]
            self.bcs[3] = args[4]
            self.rhs_solution()
            self.bc_solution()
        self.potential = np.sum(self.potentials, axis=0)


@njit(cache=True)
def series_term(X, Y, Lx, Ly, voln, rhs, n, m):
    """ Fourier series term of the analytical solution of the 2D Poisson with
    zero dirichlet bc problem """
    # Originally returning fourier_coef_2D(X, Y, Lx, Ly, voln, rhs, n, m) * np.sin(n * np.pi * X / Lx)
    # * np.sin(m * np.pi * Y / Ly) / ((n / Lx)**2 + (m / Ly)**2))
    # inlining to avoid sine calculation => speedup x2
    x_mode = np.sin(n * np.pi * X / Lx)
    y_mode = np.sin(m * np.pi * Y / Ly)
    return (4 / Lx / Ly * np.sum(x_mode * y_mode * rhs * voln)
            * x_mode * y_mode / ((n / Lx)**2 + (m / Ly)**2))


def series_term_dup(V_u, X, Y, Lx, Ly, n):
    return (fourier_coef_1D(V_u, n, X[0, :], Lx) * np.sin(n * np.pi * X / Lx)
            * np.sinh(n * np.pi * Y / Lx) / np.sinh(n * np.pi * Ly / Lx))


def series_term_ddown(V_u, X, Y, Lx, Ly, n):
    return (fourier_coef_1D(V_u, n, X[0, :], Lx) * np.sin(n * np.pi * X / Lx)
            * np.sinh(n * np.pi * (Y - Ly) / Lx) / np.sinh(- n * np.pi * Ly / Lx))


def series_term_dleft(V_u, X, Y, Lx, Ly, n):
    return (fourier_coef_1D(V_u, n, Y[:, 0], Ly) * np.sin(n * np.pi * Y / Ly)
            * np.sinh(n * np.pi * (X - Lx) / Ly) / np.sinh(- n * np.pi * Lx / Ly))


def series_term_dright(V_u, X, Y, Lx, Ly, n):
    return (fourier_coef_1D(V_u, n, Y[:, 0], Ly) * np.sin(n * np.pi * Y / Ly)
            * np.sinh(n * np.pi * X / Ly) / np.sinh(n * np.pi * Lx / Ly))
