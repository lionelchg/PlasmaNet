########################################################################################################################
#                                                                                                                      #
#                                            2D Poisson solver using numpy                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 25.09.2021                                           #
#                                                                                                                      #
########################################################################################################################

import os
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import scipy.constants as co
from pathlib import Path
from scipy import interpolate

import PlasmaNet.common.profiles as pf
from PlasmaNet.poissonscreensolver.photo_ls import PhotoLinSystem

if __name__ == '__main__':
    # Create figures directory
    fig_dir = Path('figures/photo/profiles')
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open('eval_cyl.yml') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    photo = PhotoLinSystem(cfg['photo'])

    zeros_x, zeros_y = np.zeros(photo.nnx), np.zeros(photo.nny)

    bcs = {'left':zeros_y, 'right':zeros_y, 'top':zeros_x}

    # Single gaussian
    ni0 = 1e10
    sigma_x, sigma_y = 5e-4, 5e-4
    x0, y0 = 2.0e-3, 0.0
    rhs = pf.gaussian(photo.X, photo.Y, ni0, x0, y0, sigma_x, sigma_y)

    # photo.solve(rhs, bcs)
    # photo.plot_2D(fig_dir / 'gaussian', axis='off')
    # photo.plot_2D_expanded(fig_dir / 'gaussian_expanded', axis='off')

    # # Multiple gaussians

    # rhs = (pf.gaussian(photo.X, photo.Y, 1e10, 2.2e-3, 0.0, 3e-4, sigma_y)
    #         - pf.gaussian(photo.X, photo.Y, 5e9, 1.7e-3, 0.0, sigma_x, sigma_y))

    # photo.solve(rhs, bcs)
    # photo.plot_2D(fig_dir / 'two_gaussian', axis='off')
    # photo.plot_2D_expanded(fig_dir / 'two_gaussian_expanded', axis='off')

    # Random 16 profile
    n_res_factor = 16
    nnx_lower = int(photo.nnx / n_res_factor)
    nny_lower = int(photo.nny / n_res_factor)
    x_lower, y_lower = np.linspace(photo.xmin, photo.xmax, nnx_lower), np.linspace(photo.ymin, photo.ymax, nny_lower)
    for irand in range(5):
        z_lower = 2 * np.random.random((nny_lower, nnx_lower)) - 1
        f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
        rhs = f(photo.x, photo.y) * ni0

        photo.solve(rhs, bcs)
        photo.plot_2D(fig_dir / f'random_16_{irand:d}', axis='off')
        photo.plot_2D_expanded(fig_dir / f'random_16_expanded_{irand:d}', axis='off')
