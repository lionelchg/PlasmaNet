import numpy as np
from scipy import integrate
from poissonsolver.base import BasePoisson

class PoissonAnalytical(BasePoisson):
    """ Analytical solution of the 2D Poisson problem 
    with 4 dirichlet boundaries in cartesian rectangular geometry """
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny, 
                nmax_rhs, mmax_rhs, nmax_dirichlet):
        super().__init__(xmin, xmax, nnx, ymin, ymax, nny)

        self.physical_rhs = np.zeros_like(self.X)
        self.bcs = [np.zeros(nnx), np.zeros(nnx), np.zeros(nny), np.zeros(nny)]
        self.voln = self.compute_voln()

        # Decomposition of the potentials (rhs, down, up, left, right)
        self.potentials = np.zeros((5, nny, nnx))

        self.nmax_dirichlet = nmax_dirichlet
        self.nmax_rhs = nmax_rhs
        self.mmax_rhs = mmax_rhs

    def compute_voln(self):
        """ Computes the nodal volume associated to each node (i, j) """
        voln = np.ones_like(self.X) * self.dx * self.dy
        voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
            self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2
        voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
            self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4
        return voln
    
    def init_rhs_bc(self, rhs, down, up, left, right):
        self.physical_rhs = rhs
        self.bcs[0] = down
        self.bcs[1] = up
        self.bcs[2] = left
        self.bcs[3] = right
    
    def rhs_solution(self):
        """ Solve the rhs problem """
        self.potential[0, :] = 0
        for n in range(1, self.nmax_rhs + 1):
            for m in range(1, self.mmax_rhs + 1):
                self.potentials[0] += series_term(self.X, self.Y, self.Lx, self.Ly, 
                                                        self.voln, self.physical_rhs, n, m)
        self.potentials[0] /= np.pi**2
    
    def bc_solution(self):
        """ Solve the 4 problems """
        self.potentials[1:, :] = 0
        for n in range(1, self.nmax_dirichlet + 1):
            self.potentials[1] += series_term_ddown(self.bcs[0], self.X, self.Y, self.Lx, self.Ly, n)
            self.potentials[2] += series_term_dup(self.bcs[1], self.X, self.Y, self.Lx, self.Ly, n)
            self.potentials[3] += series_term_ddown(self.bcs[2], self.Y, self.X, self.Ly, self.Lx, n)
            self.potentials[4] += series_term_dup(self.bcs[3], self.Y, self.X, self.Ly, self.Lx, n)

    def compute_sol(self):
        """ Sum of the solutions """
        self.potential = np.sum(self.potentials, axis=0)

def fourier_coef_2D(X, Y, Lx, Ly, voln, rhs, n, m):
    """ Fourier coefficient of the solution (integral over the domain) """
    return 4 / Lx / Ly * np.sum(np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly) * rhs * voln)

def series_term(X, Y, Lx, Ly, voln, rhs, n, m):
    """ Fourier series term of the analytical solution of the 2D Poisson with
    zero dirichlet bc problem """
    return (fourier_coef_2D(X, Y, Lx, Ly, voln, rhs, n, m) * np.sin(n * np.pi * X / Lx) 
                    * np.sin(m * np.pi * Y / Ly) / ((n / Lx)**2 + (m / Ly)**2))

def fourier_coef_1D(V_u, n, x, Lx):
    return integrate.simps(V_u * np.sin(n * np.pi * x / Lx), x)

def series_term_dup(V_u, X, Y, Lx, Ly, n):
    return (fourier_coef_1D(V_u, n, X[0, :], Lx) * np.sin(n * np.pi * X / Lx) * 
                np.sinh(n * np.pi * Y / Lx) / np.sinh(n * np.pi * Ly / Lx))

def series_term_ddown(V_u, X, Y, Lx, Ly, n):
    return (fourier_coef_1D(V_u, n, X[0, :], Lx) * np.sin(n * np.pi * X / Lx) * 
                np.sinh(n * np.pi * (Y - Ly) / Lx) / np.sinh(- n * np.pi * Ly / Lx))