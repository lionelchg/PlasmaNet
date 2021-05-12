########################################################################################################################
#                                                                                                                      #
#                            2D Poisson datasets for inference. 3 Dataset types are mixed.                             #
#                                     1/3 is random 1/3 is fourier and 1/3 is hills.                                   #
#                                                                                                                      #
#                                     Ekhi Ajuria, Lionel Cheng, CERFACS, 14.04.2021                                   #
#                                                                                                                      #
########################################################################################################################

import os
import time
from multiprocessing import get_context
import argparse
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from scipy import interpolate
from tqdm import tqdm

# From PlasmaNet
from PlasmaNet.poissonsolver.network import PoissonNetwork
from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
import PlasmaNet.common.profiles as pf
from PlasmaNet.poissonsolver.poisson import DatasetPoisson
from PlasmaNet.common.utils import create_dir


args = argparse.ArgumentParser(description='RHS inference dataset')
args.add_argument('-c', '--cfg', type=str, default=None,
                help='Config filename')
args.add_argument('-nn', '--nnodes', default=None, type=int,
                    help='Number of nodes in x and y directions')

args = args.parse_args()

with open(args.cfg, 'r') as yaml_stream:
    cfg = yaml.safe_load(yaml_stream)

device = cfg['device']
nits = cfg['n_entries']
n_procs = cfg['n_procs']

# Overwrite the resolution if in CLI
if args.nnodes is not None:
    cfg['poisson']['nnx'] = args.nnodes
    cfg['poisson']['nny'] = args.nnodes

# Amplitude of the RhS
ni0 = 1e11
rhs0 = ni0 * co.e / co.epsilon_0
ampl_min, ampl_max = 0.01, 1
sigma_min, sigma_max = 1e-3, 3e-3
x_middle_min, x_middle_max = 0.35e-2, 0.65e-2

class MixedDataset:
    """ Class to generate the dataset containing random, fourier and hills (1/3 each)
    """
    def __init__(self, cfg):
        # Generate poisson class and bcs
        self.poisson_ls = PoissonLinSystem(cfg['poisson'])
        zeros_x, zeros_y = np.zeros(self.poisson_ls.nnx), np.zeros(self.poisson_ls.nny)
        self.pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}
        self.case_dir = cfg['output_dir']
        self.plot_evr = cfg['plot_every']

        # Parameters inside the dataset
        self.random_factors = cfg['datasets']['random_factors']
        self.fourier_modes = cfg['datasets']['fourier_modes']
        self.fourier_decrease = cfg['datasets']['fourier_decrease']

        # Arrays containing the dataset
        self.physical_rhs = np.zeros((cfg['n_entries'], self.poisson_ls.nnx, self.poisson_ls.nny))
        self.potential = np.zeros((cfg['n_entries'], self.poisson_ls.nnx, self.poisson_ls.nny))

        # Useful parameters for dividing the dataset
        self.elements_rand = np.int((cfg['n_entries']/3)/len(self.random_factors))
        self.elements_fourier = np.int((cfg['n_entries']/3)/(len(self.fourier_decrease)*len(self.fourier_modes)))
        self.el_3 = np.int(cfg['n_entries']/3) 

    def plot_every(self, it):
        """Useful function to choose when to plot

        Args:
            plot_evr (int): interval of its for plotting
            it (int): corresponding it

        Returns:
            bool: returns a bool to decide if a plot is needed
        """

        if it%self.plot_evr==0:
            plot = True
        else:
            plot = False
        return plot

    def plot_fourier(self, poisson_ls, i, j, w, mode, dec_pow):
        """ Function to perform individual plots for the fourier dataset part

        Args:
            poisson_ls (class): poisson class
            i (int): mode number
            j (int): exponential number
            w (int): elements of each subcategory
            mode (int): number of modes taken for the Fourier case
            dec_pow (int): dumping exponent for fourier modes
        """

        if self.plot_every(w):
            case_dir_fourier = os.path.join(self.case_dir, 'fourier/fourier_{}_{}'.format(mode, dec_pow))
            fig_dir = case_dir_fourier + '/figures/'
            if not os.path.exists(fig_dir):
                create_dir(fig_dir)
            poisson_ls.potential = self.potential[w + (2*i + j)*self.elements_fourier + self.el_3]
            poisson_ls.physical_rhs = self.physical_rhs[w + (2*i + j)*self.elements_fourier + self.el_3] 
            poisson_ls.plot_2D(fig_dir + '2D_{}'.format(w), geom='xy')
            poisson_ls.plot_1D2D(fig_dir + 'full_{}'.format(w), geom='xy')

    def run_random(self, i, j, rand):
        """ Inidividual calculation for the random dataset part

        Args:
            i (int): random factor number
            j (int): individual element of subgroup
            rand (int): random factor
        """

        self.physical_rhs[j+i*self.elements_rand] = pf.random2D(self.poisson_ls.X,
                            self.poisson_ls.Y, ni0, rand) * co.e / co.epsilon_0
        case_dir_rand = os.path.join(self.case_dir, 'random/random_{}'.format(rand))
        self.poisson_ls.run_case(case_dir_rand, self.physical_rhs[j + i*self.elements_rand], self.pot_bcs, 
                            self.plot_every(j), False, j)
        self.potential[j + i*self.elements_rand] = self.poisson_ls.potential

    def run_fourier(self, poisson_ls, i, j, w, mode, dec_pow):
        """ Inidividual calculation for the random dataset part

        Args:
            i (int): random factor number
            j (int): individual element of subgroup
            rand (int): random factor
        """

        random_array = np.random.random((mode, mode))
        rhs_coefs = rhs0* (2*random_array - 1)
        rhs_coefs /= (poisson_ls.N**dec_pow + poisson_ls.M**dec_pow)
        self.potential[w + (2*i + j)*self.elements_fourier + self.el_3] = poisson_ls.pot_series(rhs_coefs)
        self.physical_rhs[w + (2*i + j)*self.elements_fourier + self.el_3] = poisson_ls.sum_series(rhs_coefs)

        self.plot_fourier(poisson_ls, i, j, w, mode, dec_pow)

    def run_hills(self, i):
        """ Individual calculation for the hills dataset part

        Args:
            i (int): individual element of subgroup
        """

        coefs = np.random.random(5)
        ampl = rhs0 * ((ampl_max - ampl_min) * coefs[0] + ampl_min)
        sigma_x = (sigma_max - sigma_min) * coefs[1] + sigma_min
        sigma_y = (sigma_max - sigma_min) * coefs[2] + sigma_min
        x_gauss = (x_middle_max - x_middle_min) * coefs[3] + x_middle_min
        y_gauss = (x_middle_max - x_middle_min) * coefs[4] + x_middle_min

        self.physical_rhs[i + 2*self.el_3] = pf.gaussian(self.poisson_ls.X, self.poisson_ls.Y, 
            ampl, x_gauss, y_gauss, sigma_x, sigma_y)

        case_dir_hills = os.path.join(self.case_dir, 'hills')
        self.poisson_ls.run_case(case_dir_hills, self.physical_rhs[i + 2*self.el_3], self.pot_bcs, 
            self.plot_every(i), False, i)
        self.potential[i + 2*self.el_3] = self.poisson_ls.potential


    def run(self, cfg):
        """General run function for the 3 dataset types

        Args:
            cfg (dict): config file loaded from yml
        """

        # Random section (takes 1/3 of the dataset)
        for i, rand in enumerate(self.random_factors):
            for j in range(self.elements_rand):
                self.run_random(i, j, rand)

        # Fourier section
        for i, mode in enumerate(self.fourier_modes):
            for j, dec_pow in enumerate(self.fourier_decrease):
                for w in range(self.elements_fourier):

                    cfg['poisson']['nmax_fourier'] = mode 
                    poisson_ls = DatasetPoisson(cfg['poisson']) 
                    self.run_fourier(poisson_ls, i, j, w, mode, dec_pow)

        # Reset poisson object
        self.poisson_ls = PoissonLinSystem(cfg['poisson'])

        # Hills dataset (takes 1/3 of the dataset)
        for i in range(self.el_3):
            self.run_hills(i)

    def save(self):
        np.save(self.case_dir + 'potential.npy', self.potential)
        np.save(self.case_dir + 'physical_rhs.npy', self.physical_rhs)


if __name__ == '__main__':

    with open(args.cfg, 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    dataset = MixedDataset(config)
    dataset.run(config)
    dataset.save()

