###########################################################################################################
#                                                                                                         #
#                                       Base class for Poisson solver                                     #
#                                                                                                         #
#                                     Lionel Cheng, CERFACS, 04.11.2020                                   #
#                                                                                                         #
###########################################################################################################

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from ..common.operators_numpy import grad, lapl
from ..common.plot import plot_ax_scalar, plot_ax_vector_arrow, plot_ax_trial_1D, plot_modes


class BasePoisson:
    """ Base class for Poisson resolution in 2D cartesian geometry
    """
    def __init__(self, cfg):
        """ Initialize BasePoisson class with the box boundaries and number of
        nodes in x and y directions in config dictionary

        :param cfg: Configuration dictionary
        :type cfg: dict
        """

        # if there is no y properties a square is assumed with same properties on
        # x and y axis
        xmin, xmax, nnx = cfg['xmin'], cfg['xmax'], cfg['nnx']
        if 'nny' in cfg:
            nny, ymin, ymax = cfg['nny'], cfg['ymin'], cfg['ymax']
        else:
            nny, ymin, ymax = nnx, xmin, xmax

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

        # Fourier series monitoring
        if 'nmax_fourier' in cfg:
            self.nmax, self.mmax = cfg['nmax_fourier'], cfg['nmax_fourier']
            if 'nmin_fourier' in cfg:
                self.nmin, self.mmin = cfg['nmin_fourier'], cfg['nmin_fourier']
            else:
                self.nmin, self.mmin = 1, 1
            self.nrange, self.mrange = np.arange(self.nmin, self.nmax + 1), np.arange(self.mmin, self.mmax + 1)
            self.N, self.M = np.meshgrid(self.nrange, self.mrange)
            self.coeffs_rhs = np.zeros(self.N.shape)
            self.coeffs_pot = np.zeros(self.N.shape)

        # Benchmark switch
        self.benchmark = False
        if "benchmark" in cfg:
            self.benchmark = cfg["benchmark"]

    def compute_voln(self):
        """ Computes the nodal volume associated to each node (j, i) """
        voln = np.ones_like(self.X) * self.dx * self.dy
        voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
            self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2
        voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
            self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4
        return voln

    @property
    def E_field(self):
        """ Electric field computation through the gradient of the potential

        :return: Electric field
        :rtype: ndarray
        """
        return - grad(self.potential, self.dx, self.dy, self.nnx, self.nny)

    @property
    def lapl(self):
        """ Laplacian computation of the potential

        :return: the Laplacian of the potential
        :rtype: ndarray
        """
        return lapl(self.potential, self.dx, self.dy, self.nnx, self.nny, r=self.R_nodes)

    def plot_2D(self, figname, geom='xy', axis='on'):
        """ Plot the potential, electric field and laplacian of the Poisson problem
        at hand in 2D fields

        :param figname: Name of the figure
        :type figname: str
        :param geom: Geometry of the problem either xy for cartesian and xr for axisymmetric, defaults to 'xy'
        :type geom: str, optional
        """

        # # For rectangular domains good plotting
        # fig = plt.figure(figsize=(8, 3))
        # gs = GridSpec(2, 4, figure=fig)
        # axes = list()
        # axes.append(fig.add_subplot(gs[0, :2]))
        # axes.append(fig.add_subplot(gs[0, 2:]))
        # axes.append(fig.add_subplot(gs[1, 1:3]))

        fig, axes = plt.subplots(ncols=3, figsize=(11, 14))

        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.potential, r'$\phi$', geom=geom)
        E = self.E_field
        plot_ax_vector_arrow(fig, axes[1], self.X, self.Y, E, r'$\mathbf{E}$', geom=geom)

        lapl_field = self.lapl
        plot_ax_scalar(fig, axes[2], self.X, self.Y, - lapl_field, r'$-\nabla^2 \phi$', geom=geom)

        # Set axes on or off
        if axis == 'off':
            for ax in axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)


        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_1D2D(self, figname, geom='xy'):
        """ Plot the potential, electric field and laplacian of the Poisson problem
        at hand in 1D cuts and 2D fields

        :param figname: Name of the figure
        :type figname: str
        :param geom: Geometry of the problem either xy for cartesian and xr for axisymmetric, defaults to 'xy'
        :type geom: str, optional
        """
        x = self.X[0, :]
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))

        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.potential, r'$\phi$', geom=geom)
        plot_ax_trial_1D(axes[0][1], x, self.potential, self.nny, '1D cuts')

        E = self.E_field
        normE = np.sqrt(E[0]**2 + E[1]**2)
        plot_ax_vector_arrow(fig, axes[1][0], self.X, self.Y, E, r'$\mathbf{E}$', geom=geom)
        plot_ax_trial_1D(axes[1][1], x, normE, self.nny, '1D cuts', ylim=[0.99 * np.min(normE), 1.01 * np.max(normE)])

        lapl_field = self.lapl
        plot_ax_scalar(fig, axes[2, 0], self.X, self.Y, - lapl_field, r'$-\nabla^2 \phi$', geom=geom)
        plot_ax_trial_1D(axes[2][1], x, -  lapl_field, self.nny, '1D cuts')

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()

    def compute_modes(self):
        """ Compute the fourier coefficients of rhs and potential """
        if self.nmax is not None:
            for i in self.nrange:
                for j in self.nrange:
                    self.coeffs_rhs[j - self.mmin, i - self.nmin] = abs(fourier_coef_2D(self.X, self.Y, self.Lx, self.Ly, self.voln, self.physical_rhs, i, j))
                    self.coeffs_pot[j - self.mmin, i - self.nmin] = self.coeffs_rhs[j - self.mmin, i - self.nmin] / np.pi**2 / ((i / self.Lx)**2 + (j / self.Ly)**2)
        else:
            print("Class is not initialized for computing modes")

    def plot_pmodes(self, figname):
        """ Plot the potential and rhs modes from 2D
        Fourier expansion """
        self.compute_modes()
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        plot_modes(ax1, self.N, self.M, self.coeffs_rhs, "RHS modes")
        ax2 = fig.add_subplot(122, projection='3d')
        plot_modes(ax2, self.N, self.M, self.coeffs_pot, "Potential modes")

        fig.tight_layout()
        fig.savefig(figname, bbox_inches='tight')
        plt.close()

    def plot_difference_modes(self, figname):
        """ Compute the fourier coefficients of rhs, target potential and output potential """

        # Start computing modes
        self.compute_modes()

        # Initialize network coeffs and coeff diference
        self.coeffs_pot_output = np.zeros_like(self.coeffs_pot)
        self.coeffs_diff = np.zeros_like(self.coeffs_pot)

        for i in self.nrange:
            for j in self.nrange:
                self.coeffs_pot_output[j - self.mmin, i - self.nmin] = abs(
                    fourier_coef_2D(self.X, self.Y, self.Lx, self.Ly, self.voln, self.potential, i, j))
                self.coeffs_diff[j - self.mmin, i - self.nmin] = abs(
                    self.coeffs_pot[j - self.mmin, i - self.nmin] -
                    self.coeffs_pot_output[j - self.mmin, i - self.nmin])

        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(141, projection='3d')
        plot_modes(ax1, self.N, self.M, self.coeffs_rhs, "RHS modes")
        ax2 = fig.add_subplot(142, projection='3d')
        plot_modes(ax2, self.N, self.M, self.coeffs_pot, "Analytical Potential modes")
        ax2.set_zlim(0,self.coeffs_pot.max())
        ax3 = fig.add_subplot(143, projection='3d')
        plot_modes(ax3, self.N, self.M, self.coeffs_pot_output, "Output Potential modes")
        ax3.set_zlim(0,self.coeffs_pot.max())
        ax4 = fig.add_subplot(144, projection='3d')
        plot_modes(ax4, self.N, self.M, self.coeffs_diff, "Difference in modes")
        ax4.set_zlim(0,self.coeffs_pot.max())

        fig.tight_layout()
        fig.savefig(figname, bbox_inches='tight')
        plt.close()

    def save(self, save_dir):
        """ Save the potential and rhs in the specified location """
        if isinstance(save_dir, Path):
            np.save(save_dir / 'physical_rhs', self.physical_rhs)
            np.save(save_dir / 'potential', self.potential)
        elif isinstance(save_dir, str):
            if save_dir[-1] != '/': save_dir += '/'
            np.save(save_dir + 'physical_rhs', self.physical_rhs)
            np.save(save_dir + 'potential', self.potential)

    def L1error_pot(self, th_potential):
        """ 1-norm error of the potential with a theoretical one

        :param th_potential: theoretical potential
        :type th_potential: ndarray
        :return: MSE error
        :rtype: float
        """

        return np.sum(np.abs(self.potential - th_potential)) / self.nnx / self.nny

    def L1error_E(self, th_E_field):
        """ 1-norm error of the electric field with a theoretical one

        :param th_potential: theoretical potential
        :type th_potential: ndarray
        :return: MSE error
        :rtype: float
        """
        eps_E = np.abs(self.E_field[0] - th_E_field[0]) + np.abs(self.E_field[1] - th_E_field[1])
        return np.sum(eps_E) / self.nnx / self.nny

    def L2error_pot(self, th_potential):
        """ MSE error of the potential with a theoretical one

        :param th_potential: theoretical potential
        :type th_potential: ndarray
        :return: MSE error
        :rtype: float
        """

        return np.sqrt(np.sum((self.potential - th_potential)**2) / self.nnx / self.nny)

    def L2error_E(self, th_E_field):
        """ MSE error of the electric field with a theoretical one

        :param th_potential: theoretical potential
        :type th_potential: ndarray
        :return: MSE error
        :rtype: float
        """
        eps_E = (self.E_field[0] - th_E_field[0])**2 + (self.E_field[1] - th_E_field[1])**2
        return np.sqrt(np.sum(eps_E) / self.nnx / self.nny)

    def Linferror_pot(self, th_potential):
        """ Infinity-norm error of the potential with a theoretical one

        :param th_potential: theoretical potential
        :type th_potential: ndarray
        :return: MSE error
        :rtype: float
        """

        return np.max(np.abs(self.potential - th_potential))

    def Linferror_E(self, th_E_field):
        """ Infinity-norm error of the electric field with a theoretical one

        :param th_potential: theoretical potential
        :type th_potential: ndarray
        :return: MSE error
        :rtype: float
        """
        eps_E = np.abs(self.E_field[0] - th_E_field[0]) + np.abs(self.E_field[1] - th_E_field[1])
        return np.max(eps_E)

    def L1_pot(self):
        """ Return the 1 norm of the potential """
        return np.sum(np.abs(self.potential)) / self.nnx / self.nny

    def L2_pot(self):
        """ Return the 2 norm of the potential """
        return np.sqrt(np.sum(self.potential**2) / self.nnx / self.nny)

    def Linf_pot(self):
        """ Return the infinity norm of the potential """
        return np.max(np.abs(self.potential))

    def L1_E(self):
        """ Return the 1 norm of the electric field """
        return np.sum(np.abs(self.E_field[0]) + np.abs(self.E_field[1])) / self.nnx / self.nny

    def L2_E(self):
        """ Return the 2 norm of the electric field """
        return np.sqrt(np.sum(self.E_field[0]**2 + self.E_field[1]**2) / self.nnx / self.nny)

    def Linf_E(self):
        """ Return the infinity norm of the electric field """
        return np.max(np.abs(self.E_field[0]) + np.abs(self.E_field[1]))


def fourier_coef_1D(V_u, n, x, Lx):
    """ Fourier coefficient of the solution of one dirichlet boundary condition
    in the square setup """
    return integrate.simps(V_u * np.sin(n * np.pi * x / Lx), x)


def fourier_coef_2D(X, Y, Lx, Ly, voln, rhs, n, m):
    """ Fourier coefficient of the solution (integral over the domain) """
    return 4 / Lx / Ly * np.sum(np.sin(n * np.pi * X / Lx) * np.sin(m * np.pi * Y / Ly) * rhs * voln)
