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
import seaborn as sns
import copy

from ..base.base_sim import BaseSim
from .scalar import compute_flux
from .chemistry import morrow
from .boundary import outlet_x, outlet_y
from ...common.plot import plot_ax_scalar, plot_ax_scalar_1D
from ...common.operators_numpy import grad
from ...common.profiles import gaussian
from ...common.utils import create_dir

from ...poissonsolver.poisson import DatasetPoisson
from ...poissonsolver.network import PoissonNetwork
from ...poissonscreensolver.photo_network import PhotoNetwork
from ...poissonscreensolver.photo_ls import PhotoLinSystem

sns.set_context('notebook', font_scale=1.0)

def V_sphere(x, r, V_0, b, E_0):
    """ Spherical electrode potential """
    d_s = np.sqrt((b + x)**2 + r**2)
    return V_0 * b / d_s - E_0 * (1 - (b / d_s)**3) * (x + b)

class StreamerMorrow(BaseSim):
    """ Streamer based on Morrow chemistry which inclues electrons, positive ions A+ and negative
    ions A-.
    """
    def __init__(self, config):
        super().__init__(config)
        # Convection speed
        self.a = np.zeros((self.ndim, self.nny, self.nnx))

        # Transport coefficients
        self.mu, self.D = np.zeros_like(self.X), np.zeros_like(self.X)

        # Background electric field and matrix construction
        for key, value in config['mesh'].items():
            config['poisson'][key] = value
        config['poisson']['geom'] = self.geom
        config['poisson']['bcs'] = 'axi'

        self.backE = config['poisson']['backE']
        # Check if spherical electrode is applied
        self.sphere_electrode = 'electrode_b' in config['poisson']
        if self.sphere_electrode:
            self.electrode_b = config['poisson']['electrode_b']
            self.electrode_V0 = config['poisson']['electrode_V0']

        # Choose the way to solve poisson equation, either classic with linear system
        # or with analytical solution (2D Fourier series)
        self.poisson_type = config['poisson']['type']

        # Initialize the poisson object depending on what is specified
        if self.poisson_type == 'lin_system':
            if self.sphere_electrode:
                self.up = V_sphere(self.x, self.ymax, self.electrode_V0, self.electrode_b, self.backE)
                self.left = V_sphere(self.xmin, self.y, self.electrode_V0, self.electrode_b, self.backE)
                self.right = V_sphere(self.xmax, self.y, self.electrode_V0, self.electrode_b, self.backE)
            else:
                self.up = - self.x * self.backE
                self.left = np.zeros_like(self.y)
                self.right = - np.ones_like(self.y) * self.backE * self.xmax
            self.bcs = {'left':self.left, 'right':self.right, 'top':self.up}
            self.poisson = DatasetPoisson(config['poisson'])

        # Network case
        if self.poisson_type == 'network':
            config['network']['eval'] = config['poisson']
            config['network']['casename'] = config['casename']
            self.poisson = PoissonNetwork(config['network'])

        # Photoionization initialization
        self.photo = 'photo' in config
        if self.photo:
            self.Sph = np.zeros_like(self.X)
            self.irate = np.zeros_like(self.X)
            self.photo_type = config['photo']['type']
            if self.photo_type == 'lin_system':
                # Copy mesh info to dict
                for key, value in config['mesh'].items():
                    config['photo'][key] = value
                # Initialize boudary conditions
                self.up_photo = np.zeros_like(self.x)
                self.left_photo = np.zeros_like(self.y)
                self.right_photo = np.zeros_like(self.y)
                self.bcs_photo = {'left':self.left_photo, 'right':self.right_photo, 'top':self.up_photo}

                self.photo_obj = PhotoLinSystem(config['photo'])
            elif self.photo_type == 'network':
                config['photo']['casename'] = config['casename']
                self.photo_obj = PhotoNetwork(config['photo'])
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
            self.x0 = config['params']['x0']
            self.nd[0, :] = gaussian(self.X, self.Y, self.n_gauss, self.x0, 0, 2e-4, 2e-4) + self.n_back
            self.nd[1, :] = gaussian(self.X, self.Y, self.n_gauss, self.x0, 0, 2e-4, 2e-4) + self.n_back
        else:
            # Scalar and Residual declaration
            self.resnd = np.zeros((3, self.nny, self.nnx))

            # Loading of densities
            self.nd = np.load(config['input']['nd'])

            self.number = int(re.search(r'_(\d+)\.npy', config['input']['ne']).group(1)) + 1
            self.dtsum = (self.number - 1) * config['output']['period'] * self.dt

        # datasets for deep-learning
        self.dl_save = config['output']['dl_save'] == 'yes'
        if self.dl_save:
            self.dl_dir = self.case_dir / 'dl_data/'
            self.dl_fig = self.dl_dir / 'figures/'
            create_dir(self.dl_dir)
            create_dir(self.dl_fig)
            self.potential_list = np.zeros((self.nit, self.nny, self.nnx))
            self.physical_rhs_list = np.zeros((self.nit, self.nny, self.nnx))
            # Compute fourier for 100 iterations
            self.dl_plot_period = int(0.1 * self.nit)
            if self.photo:
                self.irate_list = np.zeros((self.nit, self.nny, self.nnx))
                self.Sph_list = np.zeros((self.nit, self.nny, self.nnx))

        # Temporal values to store (position of positive streamer, position of negative streamer,
        # energy of the discharge)
        self.gstreamer = np.zeros((self.nit + 1, 4))
        self.gstreamer[:, 0] = np.linspace(0, self.nit * self.dt, self.nit + 1)
        self.n_middle = int(self.nnx / 2)

        # Values for 2D plotting
        self.ne_ticks, self.Emax, self.Sph_ticks = None, None, None
        if 'plot' in config['output']:
            self.ne_ticks = config['output']['plot']['ne_ticks']
            self.Emax = config['output']['plot']['Emax']
            if self.photo:
                self.Sph_ticks = config['output']['plot']['Sph_ticks']


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
        rhs_field = (self.nd[1] - self.nd[0] - self.nd[2]) * co.e / co.epsilon_0

        if self.poisson_type == 'lin_system':
            self.poisson.solve(rhs_field, self.bcs)
            self.E_field = self.poisson.E_field
        elif self.poisson_type == 'network':
            self.poisson.solve(rhs_field)
            self.E_field = self.poisson.E_field
            # Add background electric field for network as it trained from Dirichlets imposed at zero
            self.E_field[0, :, :] += self.backE

    def solve_photo(self):
        """ Solve the photoionization source term using approximation in Helmholtz equations """
        logging.info('--> Photoionization resolution')
        if self.photo_type == 'lin_system':
            self.photo_obj.solve(self.irate, self.bcs_photo)
        elif self.photo_type == 'network':
            self.photo_obj.solve(self.irate)
        self.Sph = self.photo_obj.Sph

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
                    geom=self.geom, cmap_scale='log')
            plot_ax_scalar_1D(fig, axes[5], self.X, [0.0, 0.1, 0.2], self.Sph, r"$S_{ph}$",
                    yscale='log')

        plt.tight_layout()
        fig.suptitle(f'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(self.fig_dir / f'instant_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)

        # Print the ionization rate for photoionization debugging
        if self.photo:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_ax_scalar(fig, ax, self.X, self.Y, self.irate, r"$n_e$", geom=self.geom, cmap_scale='log')
            plt.savefig(self.fig_dir / f'irate_{self.number:04d}', bbox_inches='tight')
            plt.close(fig)

        # 2D contour plots only
        if self.photo:
            fig, axes = plt.subplots(nrows=3, figsize=(6, 8))
        else:
            fig, axes = plt.subplots(nrows=2, figsize=(6, 6))

        axes = axes.reshape(-1)

        plot_ax_scalar(fig, axes[0], self.X, self.Y, self.nd[0], r"$n_e$", geom=self.geom, cmap_scale='log',
                            field_ticks=self.ne_ticks)
        plot_ax_scalar(fig, axes[1], self.X, self.Y, self.normE, r"$|\mathbf{E}|$",
                            cmap='Blues', geom=self.geom, max_value=self.Emax)
        fig.axes[0].get_xaxis().set_visible(False)
        fig.axes[0].get_yaxis().set_visible(False)
        fig.axes[1].get_xaxis().set_visible(False)
        fig.axes[1].get_yaxis().set_visible(False)
        if self.photo:
            plot_ax_scalar(fig, axes[2], self.X, self.Y, self.Sph, r"$S_{ph}$",
                    geom=self.geom, cmap_scale='log', field_ticks=self.Sph_ticks)
            fig.axes[2].get_xaxis().set_visible(False)
            fig.axes[2].get_yaxis().set_visible(False)
        plt.tight_layout()
        fig.suptitle(f'$t$ = {self.dtsum:.2e} s')
        fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(self.fig_dir / f'instant_2D_{self.number:04d}', bbox_inches='tight')
        plt.close(fig)

    def postproc(self, it):
        """ Post-processing of simulation after each iteration: save data, plot data, accumulate lists
        for deep-learning training sets as well """
        super().postproc(it)

        if self.dl_save:
            self.potential_list[it - 1, :, :] = self.poisson.potential
            self.physical_rhs_list[it - 1, :, :] = self.poisson.physical_rhs
            if it % self.dl_plot_period == 0:
                self.poisson.plot_2D(self.dl_fig + f'input_{it:05d}')

            if self.photo:
                self.irate_list[it - 1, :, :] = self.irate
                self.Sph_list[it - 1, :, :] = self.Sph

    def save(self):
        """ Save solutions. """
        np.save(self.data_dir / f'nd_{self.number:04d}', self.nd)
        np.save(self.data_dir / f'normE_{self.number:04d}', self.normE)
        np.save(self.data_dir / f'globals_{self.number:04d}', self.gstreamer)
        if self.photo:
            np.save(self.data_dir / f'Sph_{self.number:04d}', self.Sph)

    def plot_global(self):
        """ Global quantities (position of negative streamer,
        positive streamer and energy of discharge). """
        # Copy to not influence data in the .npy file that is saved
        # afterwards
        gstreamer = copy.deepcopy(self.gstreamer)

        # Scale to ns
        time = gstreamer[:, 0] / 1e-9

        # Put the first iterations to the same value of
        # later in the simulations when the heads of streamer really appeared
        gstreamer[:20, 1:3] =  gstreamer[20, 1:3]

        # Scale to mmm and microjoule
        gstreamer[:, 1:3] = gstreamer[:, 1:3] / 1e-3
        gstreamer[:, 3] = gstreamer[:, 3] / 1e-6

        # Figure creation
        fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
        axes[0].plot(time, gstreamer[:, 1], 'k--', lw=2, label='Negative streamer')
        axes[0].plot(time, gstreamer[:, 2], 'k', lw=2, label='Positive streamer')
        axes[0].set_ylabel('$x$ [mm]')
        axes[0].set_xlabel('$t$ [ns]')
        axes[0].set_ylim(np.array([self.xmin, self.xmax]) / 1e-3)
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(time, gstreamer[:, 3], 'k', lw=2)
        axes[1].set_xlabel('$t$ [ns]')
        axes[1].set_ylabel(r'E [$\mu$J]')
        axes[1].grid(True)

        fig.tight_layout()
        fig.savefig(self.fig_dir / 'globals', bbox_inches='tight')
        plt.close(fig)

    def post_temporal(self):
        """ Post-temporal processing with saving and plotting of global arrays """
        self.plot_global()
        if self.save_data:
            np.save(self.data_dir / 'globals', self.gstreamer)

        if self.dl_save:
            np.save(self.dl_dir + 'potential.npy', self.potential_list)
            np.save(self.dl_dir + 'physical_rhs.npy', self.physical_rhs_list)
            if self.photo:
                np.save(self.dl_dir + 'Sph.npy', self.Sph_list)
                np.save(self.dl_dir + 'irate.npy', self.irate_list)

    @classmethod
    def run(cls, config):
        """ Main function containing initialisation, temporal loop and outputs. Takes a config dict as input. """

        sim = cls(config)

        # Print information
        sim.print_init()

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

        # Post-processing and saving of specified variables
        sim.post_temporal()

def main():
    args = argparse.ArgumentParser(description='Streamer run')
    args.add_argument('-c', '--config', type=str,
                        help='Config filename', required=True)
    args = args.parse_args()

    with open(args.config, 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    StreamerMorrow.run(cfg)

if __name__ == '__main__':
    main()