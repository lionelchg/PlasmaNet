import numpy as np
import matplotlib.pyplot as plt

from cfdsolver.base.base_plot import plot_ax_scalar
from cfdsolver.scalar.init import gaussian, two_gaussians, random
from cfdsolver.utils import create_dir

if __name__ == '__main__':
    fig_dir = 'figures/'
    create_dir(fig_dir)

    xmin, xmax, nnx = 0, 0.01, 101
    ymin, ymax, nny = 0, 0.01, 101
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
    X, Y = np.meshgrid(x, y)
    profiles = {}
    profiles['gaussian'] = gaussian(X, Y, 1.0, 0.5e-2, 0.5e-2, 1.0e-3, 1.0e-3)
    profiles['off_gaussian'] = gaussian(X, Y, 1.0, 0.4e-2, 0.35e-2, 1.0e-3, 1.0e-3)
    profiles['two_gaussians'] = two_gaussians(X, Y, 1.0, 0.4e-2, 0.5e-2, 1.0e-3, 1.0e-3, 0.6e-2, 0.5e-2, 1.0e-3, 1.0e-3)
    profiles['random_4'] = random(X, Y, 1.0, 4, 3.0e-3)
    profiles['random_8'] = random(X, Y, 1.0, 8, 3.0e-3)
    profiles['random_16'] = random(X, Y, 1.0, 16, 3.0e-3)

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    for i, profile in enumerate(profiles.values()):
        plot_ax_scalar(fig, axes[i // 3][i % 3], X, Y, profile, '', geom='xy', cbar=False)
        axes[i // 3][i % 3].set_xticks([])
        axes[i // 3][i % 3].set_yticks([])
    fig.savefig(fig_dir + 'profiles', bbox_inches='tight')
