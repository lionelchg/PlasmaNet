import numpy as np
import scipy.constants as co
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
from poissonsolver.operators import grad, lapl
from poissonsolver.plot import plot_ax_scalar, plot_ax_vector_arrow, plot_ax_trial_1D, plot_modes

class BasePoisson:
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny, nmax=None):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.Lx, self.Ly = xmax - xmin, ymax - ymin
        self.dx, self.dy = (xmax - xmin) / (nnx - 1), (ymax - ymin) / (nny - 1)
        self.x, self.y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
        self.nnx, self.nny = nnx, nny

        # Mesh attributes
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.voln = self.compute_voln()

        # Sum of the potentials
        self.potential = np.zeros_like(self.X)
        self.physical_rhs = np.zeros_like(self.X)

        # Radius of the nodes for axisymmetric configuration
        self.R_nodes = None

        if nmax is not None:
            self.nmax, self.mmax = nmax, nmax
            self.nrange, self.mrange = np.arange(1, self.nmax + 1), np.arange(1, self.mmax + 1)
            self.N, self.M = np.meshgrid(self.nrange, self.mrange)
            self.coeffs_rhs = np.zeros(self.N.shape)
            self.coeffs_pot = np.zeros(self.N.shape)

    def compute_voln(self):
        """ Computes the nodal volume associated to each node (i, j) """
        voln = np.ones_like(self.X) * self.dx * self.dy
        voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
            self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2
        voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
            self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4
        return voln

    @property
    def E_field(self):
        return - grad(self.potential, self.dx, self.dy, self.nnx, self.nny)
    
    @property
    def lapl(self):
        return lapl(self.potential, self.dx, self.dy, self.nnx, self.nny, r=self.R_nodes)


    def plot_2D(self, figname, axi=False):

        fig, axes = plt.subplots(ncols=3, figsize=(11, 14))

        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.potential, 'Potential', axi=axi)

        E = self.E_field
        plot_ax_vector_arrow(fig, axes[1], self.X, self.Y, E, 'Electric field', axi=axi)

        lapl_field = self.lapl
        plot_ax_scalar(fig, axes[2], self.X, self.Y, - lapl_field, '- Laplacian', axi=axi)

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()

    def plot_1D2D(self, figname, axi=False):
        # 1D vector
        x = self.X[0, :]

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 14))

        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.potential, 'Potential', axi=axi)
        plot_ax_trial_1D(axes[0][1], x, self.potential, self.nny, '1D cuts')

        E = self.E_field
        normE = np.sqrt(E[0]**2 + E[1]**2)
        plot_ax_vector_arrow(fig, axes[1][0], self.X, self.Y, E, 'Electric field', axi=axi)
        plot_ax_trial_1D(axes[1][1], x, normE, self.nny, '1D cuts', ylim=[0.99 * np.min(normE), 1.01 * np.max(normE)])

        lapl_field = self.lapl
        plot_ax_scalar(fig, axes[2, 0], self.X, self.Y, - lapl_field, '- Laplacian', axi=axi)
        plot_ax_trial_1D(axes[2][1], x, -  lapl_field, self.nny, '1D cuts')

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()

    def compute_modes(self):
        """ Compute the fourier coefficients of rhs and potential """
        if self.nmax is not None:
            for i in self.nrange:
                for j in self.nrange:
                    self.coeffs_rhs[j - 1, i - 1] = fourier_coef_2D(self.X, self.Y, self.Lx, self.Ly, self.voln, self.physical_rhs, i, j)
                    self.coeffs_pot[j - 1, i - 1] = self.coeffs_rhs[j - 1, i - 1] / np.pi**2 / ((i / self.Lx)**2 + (j / self.Ly)**2)
        else:
            print("Class is not initialized for computing modes")
    
    def plot_pmodes(self, figname):
        """ Plot the potential and rhs modes from 2D
        Fourier expansion """
        self.compute_modes()
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        plot_modes(ax1, self.N, self.M, abs(self.coeffs_rhs), "RHS modes")
        ax2 = fig.add_subplot(122, projection='3d')
        plot_modes(ax2, self.N, self.M, abs(self.coeffs_pot), "Potential modes")

        fig.tight_layout()
        fig.savefig(figname, bbox_inches='tight')
        plt.close()


def fourier_coef_1D(V_u, n, x, Lx):
    """ Fourier coefficient of the solution of one dirichlet boundary condition
    in the square setup """
    return integrate.simps(V_u * np.sin(n * np.pi * x / Lx), x)

def fourier_coef_2D(X, Y, Lx, Ly, voln, rhs, n, m):
    """ Fourier coefficient of the solution (integral over the domain) """
    return 4 / Lx / Ly * np.sum(np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly) * rhs * voln)
