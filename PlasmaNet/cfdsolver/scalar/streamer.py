########################################################################################################################
#                                                                                                                      #
#                                    Scalar transport equations related routines                                       #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as co
import re
import argparse
import yaml
import logging

from scipy.sparse.linalg import spsolve

from ..base.basesim import BaseSim
from .scalar import compute_flux
from .chemistry import morrow
from .photo import photo_axisym, A_j_two, A_j_three, lambda_j_two, lambda_j_three
from .boundary import outlet_x, outlet_y
from ...common.plot import plot_ax_scalar, plot_ax_scalar_1D
from ...common.operators_numpy import grad
from ...common.profiles import gaussian

from ...poissonsolver.linsystem import matrix_axisym, dirichlet_bc_axi

class StreamerMorrow(BaseSim):
    def __init__(self, config):
        super().__init__(config)
        # Convection speed
        self.a = np.zeros((self.ndim, self.nny, self.nnx))

        # Transport coefficients
        self.mu, self.D = np.zeros_like(self.X), np.zeros_like(self.X)

        # Background electric field and matrix construction
        self.backE = config['poisson']['backE']
        self.up = - self.x * self.backE
        self.left = np.zeros_like(self.y)
        self.right = - np.ones_like(self.y) * self.backE * self.xmax
        self.scale = self.dx * self.dy
        self.mat_poisson = matrix_axisym(self.dx, self.dy, self.nnx,
                                         self.nny, self.R_nodes, self.scale)

        # Photoionization initialization
        self.photo = config['params']['photoionization'] != 'no'
        if self.photo:
            self.photo_model = config['params']['photoionization']
            # Pressure in Torr
            self.pO2 = 150

            # Boundary conditions
            self.up_photo = np.zeros_like(self.x)
            self.left_photo = np.zeros_like(self.y)
            self.right_photo = np.zeros_like(self.y)

            self.mats_photo = []
            self.irate = np.zeros_like(self.X)
            self.Sph = np.zeros_like(self.X)

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
        else:
            self.irate = None

        # Read or initialize solution
        self.input_fn = config['input']
        if self.input_fn == 'none':
            self.number = 1

            self.nd = np.zeros((3, self.nny, self.nnx))  # Electrons, positive ions, negative ions
            self.resnd = np.zeros((3, self.nny, self.nnx))
            # Gaussian initialization for the electrons and positive ions
            self.n_back = config['params']['n_back']
            self.n_gauss = config['params']['n_gauss']
            self.nd[0, :] = gaussian(self.X, self.Y, self.n_gauss, 2e-3, 0, 2e-4, 2e-4) + self.n_back
            self.nd[1, :] = gaussian(self.X, self.Y, self.n_gauss, 2e-3, 0, 2e-4, 2e-4) + self.n_back
        else:
            # Scalar and Residual declaration
            self.resnd = np.zeros((3, self.nny, self.nnx))

            # Loading of densities
            self.nd = np.load(config['input']['nd'])

            self.number = int(re.search('_(\d+)\.npy', config['input']['ne']).group(1)) + 1
            self.dtsum = (self.number - 1) * config['output']['period'] * self.dt

        # Temporal values to store (position of positive streamer, position of negative streamer, energy of the discharge)
        self.gstreamer = np.zeros((self.nit + 1, 4))
        self.gstreamer[:, 0] = np.linspace(0, self.nit * self.dt, self.nit + 1)
        self.n_middle = int(self.nnx / 2)

    def print_init(self):
        """ Print header to sum up the parameters. """
        logging.info(f'Number of nodes: nnx = {self.nnx:d} -- nny = {self.nny:d}')
        logging.info(f'Bounding box: ({self.xmin:.1e}, {self.ymin:.1e}), ({self.xmax:.1e}, {self.ymax:.1e})')
        logging.info(f'dx = {self.dx:.2e} -- dy = {self.dy:.2e} -- Timestep = {self.dt:.2e}')
        logging.info('------------------------------------')
        if self.input_fn == 'none':
            logging.info('Start of simulation')
        else:
            logging.info('Restart of simulation')
        logging.info('------------------------------------')
        logging.info('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    def solve_poisson(self):
        """ Solve the Poisson equation in axisymmetric configuration. """
        self.physical_rhs = (self.nd[1] - self.nd[0] - self.nd[2]).reshape(-1) * co.e / co.epsilon_0
        self.rhs = - self.physical_rhs * self.scale
        dirichlet_bc_axi(self.rhs, self.nnx, self.nny, self.up, self.left, self.right)
        self.potential = spsolve(self.mat_poisson, self.rhs).reshape(self.nny, self.nnx)
        self.E_field = - grad(self.potential, self.dx, self.dy, self.nnx, self.nny)
        self.physical_rhs = self.physical_rhs.reshape((self.nny, self.nnx))
    
    def solve_photo(self):
        """ Solve the photoionization source term using approximation in Helmholtz equations """
        self.Sph[:] = 0
        logging.info('--> Photoionization resolution')
        if self.photo_model == 'two':
            for i in range(2):
                rhs = - self.irate.reshape(-1) * A_j_two[i] * self.pO2**2 * self.scale
                dirichlet_bc_axi(rhs, self.nnx, self.nny, self.up_photo, self.left_photo, self.right_photo)
                self.Sph += spsolve(self.mats_photo[i], rhs).reshape(self.nny, self.nnx)
        elif self.photo_model == 'three':
            for i in range(3):
                rhs = - self.irate.reshape(-1) * A_j_three[i] * self.pO2**2 * self.scale
                dirichlet_bc_axi(rhs, self.nnx, self.nny, self.up_photo, self.left_photo, self.right_photo)
                self.Sph += spsolve(self.mats_photo[i], rhs).reshape(self.nny, self.nnx)

    def compute_chemistry(self, it):
        """ Apply chemistry from Morrow et. al. with or without photoionization. """
        morrow(self.mu, self.D, self.E_field, self.nd, self.resnd, self.nnx, self.nny, self.voln, irate=self.irate)

        if self.photo:
            self.resnd[0] -= self.Sph * self.voln
            self.resnd[1] -= self.Sph * self.voln

    def compute_residuals(self):
        """ Compute convective and diffusive fluxes. """
        self.a = - self.mu * self.E_field
        self.diff_flux = self.D * grad(self.nd[0], self.dx, self.dy, self.nnx, self.nny)

        # Loop on the cells to compute the interior flux and update residuals
        compute_flux(self.resnd[0], self.a, self.nd[0], self.diff_flux, self.sij, self.ncx, self.ncy, r=self.y)
        # Boundary conditions
        outlet_y(self.resnd[0], self.a, self.nd[0], self.diff_flux, self.dx, -1, r=np.max(self.Y))
        outlet_x(self.resnd[0], self.a, self.nd[0], self.diff_flux, self.dy, 0, r=self.Y)
        outlet_x(self.resnd[0], self.a, self.nd[0], self.diff_flux, self.dy, -1, r=self.Y)

    def update_res(self):
        """ Update residual. """
        for i in range(3):
            self.nd[i] -= self.resnd[i] * self.dt / self.voln

    def global_prop(self, it):
        """ Compute global propagation of the streamers and discharge power."""
        E_field = self.E_field
        x = self.x
        self.normE = np.sqrt(E_field[0, :, :]**2 + E_field[1, :, :]**2) 
        self.normE_ax = self.normE[0, :]  # compute norm on the axis
        # maximal electric field position in the first half of the domain
        # ie. left streamer position
        self.gstreamer[it, 1] = x[np.argmax(self.normE_ax[:self.n_middle])]  
        # right streamer position
        self.gstreamer[it, 2] = x[self.n_middle + np.argmax(self.normE_ax[self.n_middle:])]
        # instantaneous discharge power
        self.gstreamer[it, 3] = self.gstreamer[it - 1, 3] + co.e * self.dt * np.sum(self.nd[0] * self.mu * self.normE * self.voln)

    def plot(self):
        """ Execute plots. """
        if self.photo:
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10))
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

        axes = axes.reshape(-1)

        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.nd[0], r"$n_e$", geom=self.geom, cmap_scale='log')
        plot_ax_scalar_1D(fig, axes[1], self.X, [0.0, 0.1, 0.2], self.nd[0], r"$n_e$", yscale='log')
        plot_ax_scalar(fig, axes[2], self.X, self.Y, self.normE, r"$|\mathbf{E}|$", geom=self.geom)
        plot_ax_scalar_1D(fig, axes[3], self.X, [0.0, 0.1, 0.2], self.normE, r"$|\mathbf{E}|$")
        if self.photo:
            plot_ax_scalar(fig, axes[4], self.X, self.Y, self.Sph, r"$S_{ph}$", 
                    geom=self.geom, cmap_scale='log', field_ticks=[1e23, 1e26, 1e29])
            plot_ax_scalar_1D(fig, axes[5], self.X, [0.0, 0.1, 0.2], self.Sph, r"$S_{ph}$", 
                    yscale='log', ylim=[1e23, 1e29])

        plt.tight_layout()
        fig.suptitle(f'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(self.fig_dir + 'instant_%04d' % self.number, bbox_inches='tight')
        plt.close(fig)

    def save(self):
        """ Save solutions. """
        np.save(self.data_dir + f'nd_{self.number:04d}', self.nd)

    def plot_global(self):
        """ Global quantities (position of negative streamer, 
        positive streamer and energy of discharge). """
        gstreamer = self.gstreamer
        time = gstreamer[:, 0] / 1e-9
        gstreamer[:, 1:3] = gstreamer[:, :2] / 1e-3
        gstreamer[:, 3] = gstreamer[:, 3] / 1e-6
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        axes[0].plot(time, gstreamer[:, 1], label='Negative streamer')
        axes[0].plot(time, gstreamer[:, 2], label='Positive streamer')
        axes[0].set_ylabel('$x$ [mm]')
        axes[0].set_xlabel('$t$ [ns]')
        axes[0].set_ylim(np.array([self.xmin, self.xmax]) / 1e-3)
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time, gstreamer[:, 3])
        axes[1].set_xlabel('$t$ [ns]')
        axes[1].set_ylabel('E [$\mu$J]')
        axes[1].grid(True)

        fig.savefig(self.fig_dir + 'globals', bbox_inches='tight')
        plt.close(fig)

    @classmethod
    def run(cls, config):
        """ Main function containing initialisation, temporal loop and outputs. Takes a config dict as input. """

        sim = cls(config)

        # Print information
        sim.print_init()

        nit, nny, nnx = sim.nit, sim.nny, sim.nnx

        if config['output']['dl_save'] == 'yes':
            potential_list = np.zeros((nit, nny, nnx))
            physical_rhs_list = np.zeros((nit, nny, nnx))
            if sim.photo:
                Sph_list = np.zeros((nit, nny, nnx))
                irate_list = np.zeros((nit, nny, nnx))

        # Iterations
        for it in range(1, sim.nit + 1):
            sim.dtsum += sim.dt

            # Solve poisson equation from charge distribution
            sim.solve_poisson()

            # Update of the residual to zero
            sim.resnd[:] = 0

            # Solve photoionization if activated
            if sim.photo and it % 10 == 1: sim.solve_photo()

            # Compute the chemistry source terms with or without photo
            sim.compute_chemistry(it)

            # Compute transport terms
            sim.compute_residuals()

            # Apply residuals
            sim.update_res()

            # Post processing of macro values
            sim.global_prop(it)

            # General post processing
            sim.postproc(it)

            if config['output']['dl_save'] == 'yes':
                potential_list[it - 1, :, :] = sim.potential + sim.backE * sim.X
                physical_rhs_list[it - 1, :, :] = sim.physical_rhs
                if sim.photo:
                    irate_list[it - 1, :, :] = sim.irate
                    Sph_list[it - 1, :, :] = sim.Sph

        sim.plot_global()
        if sim.save_data: np.save(sim.data_dir + 'globals', sim.gstreamer)

        if config['output']['dl_save'] == 'yes':
            np.save(config['output']['folder'] + config['casename'] + 'potential.npy', potential_list)
            np.save(config['output']['folder'] + config['casename'] + 'physical_rhs.npy', physical_rhs_list)
            if sim.photo:
                np.save(config['output']['folder'] + config['casename'] + 'Sph.npy', Sph_list)
                np.save(config['output']['folder'] + config['casename'] + 'irate.npy', irate_list)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Streamer run')
    args.add_argument('-c', '--config', type=str,
                        help='Config filename', required=True)
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    StreamerMorrow.run(cfg)