########################################################################################################################
#                                                                                                                      #
#                                        2D Poisson analytical solution tests                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 10.03.2020                                           #
#                                                                                                                      #
########################################################################################################################
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
import yaml
import scipy.constants as co
from scipy import interpolate
from pathlib import Path
from time import perf_counter, time

from PlasmaNet.poissonsolver.poisson import PoissonLinSystem
from PlasmaNet.poissonsolver.analytical import PoissonAnalytical
from PlasmaNet.poissonsolver.network import PoissonNetwork
import PlasmaNet.common.profiles as pf

def precision_runs(poisson: PoissonLinSystem, poisson_nn: PoissonNetwork,
    cfg: dict, fig_dir, physical_rhs, nsolves, nmax_modes):
    # Creation of figures directory
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Number of solves
    print(f'{nsolves:d} solves')

    # linear system solve and solution norm
    tstart_ls = perf_counter()
    for _ in range(nsolves):
        poisson.solve(physical_rhs, pot_bcs)
    tend_ls = perf_counter()
    print(f'Linear system resolution time: {(tend_ls - tstart_ls) / nsolves:.3e} s')
    norm_pot_ref = dict()
    norm_E_ref = dict()
    norm_pot_ref['L1'], norm_pot_ref['L2'], norm_pot_ref['Linf'] = poisson.L1_pot(), poisson.L2_pot(), poisson.Linf_pot()
    norm_E_ref['L1'], norm_E_ref['L2'], norm_E_ref['Linf'] = poisson.L1_E(), poisson.L2_E(), poisson.Linf_E()
    poisson.plot_1D2D(fig_dir / 'ls')

    # neural network
    errors_nn = {'L1': {}, 'L2': {}, 'Linf': {}}
    tstart_nn = perf_counter()
    for _ in range(nsolves):
        poisson_nn.solve(physical_rhs)
    tend_nn = perf_counter()
    print(f'Network resolution time: {(tend_nn - tstart_nn) / nsolves:.3e} s')
    for key in errors_nn:
        errors_nn[key] = {'potential':100 * getattr(poisson_nn, f'{key}error_pot')(poisson.potential) / norm_pot_ref[key],
                'E_field': getattr(poisson_nn, f'{key}error_E')(poisson.E_field) / norm_E_ref[key]}
        print(f'Potential {key} error (%): {errors_nn[key]["potential"]:.3f}')
        print(f'Electric field {key} error (%): {errors_nn[key]["E_field"]:.3f}')

    # Declaration of class Poisson Analytical
    poisson_series = PoissonAnalytical(cfg)

    # Variation of number of modes
    error_modes = np.zeros((6, len(nmax_modes)))

    for inmax, nmax in enumerate(nmax_modes):
        # Set the new values for analytical solution
        poisson_series.nmax_rhs = nmax
        poisson_series.mmax_rhs = nmax
        
        # Solve rhs problem
        tstart_an = perf_counter()
        for _ in range(nsolves):
            poisson_series.compute_sol(physical_rhs)
        tend_an = perf_counter()
        print(f'{nmax:d}^2 modes Fourier series resolution time: {(tend_an - tstart_an) / nsolves:.3e} s')
        poisson_series.plot_1D2D(fig_dir / f'{nmax}_modes')

        # Compute error in percentage
        error_modes[0, inmax] = 100 * poisson_series.L1error_pot(poisson.potential) / norm_pot_ref['L1']
        error_modes[1, inmax] = 100 * poisson_series.L1error_E(poisson.E_field) / norm_E_ref['L1']
        error_modes[2, inmax] = 100 * poisson_series.L2error_pot(poisson.potential) / norm_pot_ref['L2']
        error_modes[3, inmax] = 100 * poisson_series.L2error_E(poisson.E_field) / norm_E_ref['L2']
        error_modes[4, inmax] = 100 * poisson_series.Linferror_pot(poisson.potential) / norm_pot_ref['Linf']
        error_modes[5, inmax] = 100 * poisson_series.Linferror_E(poisson.E_field) / norm_E_ref['Linf']
    
    # Plot the errors
    plot_error(error_modes[0], error_modes[1], fig_dir / 'errors_L1')
    plot_error(error_modes[2], error_modes[3], fig_dir / 'errors_L2')
    plot_error(error_modes[4], error_modes[5], fig_dir / 'errors_Linf')

def plot_error(pot_error, E_error, figname):
    """ Plot the potential and electric field errors """
    fig, ax = plt.subplots()
    ax.plot(nmax_modes, pot_error, 'k', label='Potential')
    ax.plot(nmax_modes, E_error, 'b', label='Electric field')
    ax.set_xlabel('$N$')
    ax.set_ylabel('Error (%)')
    ax.grid(True)
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(figname)

if __name__ == '__main__':
    fig_dir = Path('figures/precision/')
    fig_dir.mkdir(parents=True, exist_ok=True)
        
    with open('poisson_ls_xy.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    poisson = PoissonLinSystem(cfg)

    zeros_x, zeros_y = np.zeros(poisson.nnx), np.zeros(poisson.nny)
    ones_x, ones_y = np.ones(poisson.nnx), np.ones(poisson.nny)
    pot_bcs = {'left':zeros_y, 'right':zeros_y, 'bottom':zeros_x, 'top':zeros_x}

    # Network solver
    with open('eval_unet5-rf200.yml') as yaml_stream:
        cfg_nn = yaml.safe_load(yaml_stream)
    poisson_nn = PoissonNetwork(cfg_nn['network'])

    # modes studied
    nmax_modes = [2, 4, 6, 8, 10, 12, 15, 20]

    # creating the rhs
    ni0 = 1e16
    sigma_x, sigma_y = 1e-3, 1e-3
    x0, y0 = 0.6e-2, 0.5e-2
    x01, y01 = 0.4e-2, 0.5e-2    
    params_gauss = [ni0, x0, y0, sigma_x, sigma_y, ni0, x01, y01, sigma_x, sigma_y]
    # Two gaussians rhs
    physical_rhs = pf.gaussians(poisson.X, poisson.Y, params_gauss) * co.e / co.epsilon_0
    # run analytical solutions for physical_rhs
    precision_runs(poisson, poisson_nn, cfg, fig_dir / 'two_gaussians', physical_rhs, 5, nmax_modes)

    # creating random rhs
    nmax_modes = [2, 4, 6, 8, 10, 12, 14, 15, 20, 30, 40, 50]
    n_res_factor = 8
    nnx_lower = int(poisson.nnx / n_res_factor)
    nny_lower = int(poisson.nny / n_res_factor)
    z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
    x_lower, y_lower = np.linspace(poisson.xmin, poisson.xmax, nnx_lower), np.linspace(poisson.ymin, poisson.ymax, nny_lower)
    f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
    physical_rhs = f(poisson.x, poisson.y) * ni0 * co.e / co.epsilon_0
    # run analytical solutions for physical_rhs
    precision_runs(poisson, poisson_nn, cfg, fig_dir / 'random_8', physical_rhs, 5, nmax_modes)