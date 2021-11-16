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


class BasePhoto:
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

        # Sum of the Sphs
        self.Sph = np.zeros_like(self.X)
        self.ioniz_rate = np.zeros_like(self.X)

        # Radius of the nodes for axisymmetric configuration
        self.R_nodes = None

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

    def plot_2D(self, figname, geom='xr', axis='on'):
        """ Plot the Sph, electric field and laplacian of the Poisson problem
        at hand in 2D fields

        :param figname: Name of the figure
        :type figname: str
        :param geom: Geometry of the problem either xy for cartesian and xr for axisymmetric, defaults to 'xy'
        :type geom: str, optional
        """

        fig, axes = plt.subplots(ncols=2, figsize=(10, 6))

        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.Sph, r'$S_{ph}$', geom=geom)

        plot_ax_scalar(fig, axes[1], self.X, self.Y, self.ioniz_rate, r'$I$', geom=geom)

        # Set axes on or off
        if axis == 'off':
            for ax in axes.reshape(-1):
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)


        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_2D_variable(self, figname, Sph, ionization_rate, lamb, geom='xr', axis='on'):
        """ Plot the Sph, electric field and laplacian of the Poisson problem
        at hand in 2D fields. Add chosen lambda as title

        :param figname: Name of the figure
        :type figname: str
        :param Sph: Sph of the field
        :type Sph: np.array
        :param ionization_rate: ionization_rate of the field
        :type ionization_rate: np.array
        :param lamb: lambda value for the equation
        :type lamb: float
        :param geom: Geometry of the problem either xy for cartesian and xr for axisymmetric, defaults to 'xy'
        :type geom: str, optional
        """

        fig, axes = plt.subplots(ncols=2, figsize=(10, 6))

        plot_ax_scalar(fig, axes[0], self.X, self.Y, Sph, r'$S_{ph}$', geom=geom)

        plot_ax_scalar(fig, axes[1], self.X, self.Y, ionization_rate, r'$I$', geom=geom)

        # Set axes on or off
        if axis == 'off':
            for ax in axes.reshape(-1):
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        fig.suptitle(f'Lambda = {lamb}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_2D_expanded(self, figname, geom='xr', axis='on'):
        """ Plot the Sph, electric field and laplacian of the Poisson problem
        at hand in 2D fields

        :param figname: Name of the figure
        :type figname: str
        :param geom: Geometry of the problem either xy for cartesian and xr for axisymmetric, defaults to 'xy'
        :type geom: str, optional
        """

        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(6, 3))

        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.Sph, r'$S_{ph}$', geom=geom)

        plot_ax_scalar(fig, axes[0][1], self.X, self.Y, self.ioniz_rate, r'$I$', geom=geom)

        plot_ax_scalar(fig, axes[1][0], self.X, self.Y, self.Sphj1, rf'$S_{{ph}}^1$', geom=geom)
        plot_ax_scalar(fig, axes[1][1], self.X, self.Y, self.Sphj2, rf'$S_{{ph}}^2$', geom=geom)

        # Set axes on or off
        if axis == 'off':
            for ax in axes.reshape(-1):
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)


        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, dpi=200, bbox_inches='tight')
        plt.close()

    def plot_1D2D(self, figname, geom='xy'):
        """ Plot the Sph, electric field and laplacian of the Poisson problem
        at hand in 1D cuts and 2D fields

        :param figname: Name of the figure
        :type figname: str
        :param geom: Geometry of the problem either xy for cartesian and xr for axisymmetric, defaults to 'xy'
        :type geom: str, optional
        """
        x = self.X[0, :]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 12))

        plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.Sph, r'$\phi$', geom=geom)
        plot_ax_trial_1D(axes[0][1], x, self.Sph, self.nny, '1D cuts')

        plot_ax_scalar(fig, axes[1, 0], self.X, self.Y, self.ioniz_rate, r'$-\nabla^2 \phi$', geom=geom)
        plot_ax_trial_1D(axes[1][1], x, self.ioniz_rate, self.nny, '1D cuts')

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()


    def save(self, save_dir):
        """ Save the Sph and rhs in the specified location """
        if isinstance(save_dir, Path):
            np.save(save_dir / 'ioniz_rate', self.ioniz_rate)
            np.save(save_dir / 'Sph', self.Sph)
        elif isinstance(save_dir, str):
            if save_dir[-1] != '/': save_dir += '/'
            np.save(save_dir + 'ioniz_rate', self.ioniz_rate)
            np.save(save_dir + 'Sph', self.Sph)

    def L1error_sph(self, th_Sph):
        """ 1-norm error of the Sph with a theoretical one

        :param th_Sph: theoretical Sph
        :type th_Sph: ndarray
        :return: MSE error
        :rtype: float
        """

        return np.sum(np.abs(self.Sph - th_Sph)) / self.nnx / self.nny

    def L2error_sph(self, th_Sph):
        """ MSE error of the Sph with a theoretical one

        :param th_Sph: theoretical Sph
        :type th_Sph: ndarray
        :return: MSE error
        :rtype: float
        """

        return np.sqrt(np.sum((self.Sph - th_Sph)**2) / self.nnx / self.nny)

    def Linferror_sph(self, th_Sph):
        """ Infinity-norm error of the Sph with a theoretical one

        :param th_Sph: theoretical Sph
        :type th_Sph: ndarray
        :return: MSE error
        :rtype: float
        """

        return np.max(np.abs(self.Sph - th_Sph))

    def L1_sph(self):
        """ Return the 1 norm of the Sph """
        return np.sum(np.abs(self.Sph)) / self.nnx / self.nny

    def L2_sph(self):
        """ Return the 2 norm of the Sph """
        return np.sqrt(np.sum(self.Sph**2) / self.nnx / self.nny)

    def Linf_sph(self):
        """ Return the infinity norm of the Sph """
        return np.max(np.abs(self.Sph))
