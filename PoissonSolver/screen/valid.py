from PlasmaNet.poissonscreensolver.photo_ls import PhotoLinSystem
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from scipy.sparse.linalg import spsolve
import yaml

from PlasmaNet.common.profiles import gaussian
from PlasmaNet.common.plot import plot_ax_scalar, plot_ax_scalar_1D
from PlasmaNet.poissonsolver.linsystem import impose_dirichlet
from PlasmaNet.poissonscreensolver.photo import A_j_two, lambda_j_two, photo_axisym

def plot_Sph(photo, X, R, Sph, figname):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 8))
    plot_ax_scalar(fig, axes[0], X, R, Sph, 'Sph 2D', cmap_scale='log', geom='xr', field_ticks=[1e23, 1e26, 1e29])
    for j in range(photo.jtot):
        axes[1].plot(photo.x, eval(f'photo.Sphj{j+1:d}[0, :]'), label=f'Sph_{j:d}')
    plot_ax_scalar_1D(fig, axes[1], X, [0], Sph, "Sph 1D cuts", yscale='log', ylim=[1e23, 1e29])
    plt.savefig(figname, bbox_inches='tight')


def plot_Sph_irate(X, R, Sph, irate, figname):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 5))
    plot_ax_scalar(fig, axes[0], X, R, irate, 'Ionization rate', geom='xr')
    plot_ax_scalar(fig, axes[1], X, R, Sph, 'Sph', geom='xr')
    plt.savefig(figname, bbox_inches='tight')

if __name__ == '__main__':
    # Figures directory
    fig_dir = Path('figures/photo/valid')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Read config file
    with open('valid.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    photo_models = ['two', 'three']

    for photo_model in photo_models:
        cfg['photo']['photo_model'] = photo_model

        # Photo class
        photo = PhotoLinSystem(cfg['photo'])

        # creating the rhs
        I0 = 3.5e28
        sigma_x, sigma_r = 1e-4, 1e-4
        x0, r0 = 1e-3, 0
        rhs = np.zeros_like(photo.ioniz_rate)
        pO2 = 150
        I = gaussian(photo.X, photo.Y, I0, x0, r0, sigma_x, sigma_r)

        # Boundary conditions
        up = np.zeros_like(photo.x)
        left = np.zeros_like(photo.y)
        right = np.zeros_like(photo.y)
        bcs = {'left':left, 'right':right, 'top':up}

        # Solve photoionization
        photo.solve(I, bcs)

        # Plot
        plot_Sph(photo, photo.X, photo.Y, photo.Sph, fig_dir / f'Sph_{photo.photo_model}')
        plot_Sph_irate(photo.X, photo.Y, photo.Sph, I, fig_dir / f'Sph_irate_{photo.photo_model}')


