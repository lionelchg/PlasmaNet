###########################################################################################################
#                                                                                                         #
#                                       Profiles for initialisation                                       #
#                                                                                                         #
#                                     Lionel Cheng, CERFACS, 28.02.2021                                   #
#                                                                                                         #
###########################################################################################################

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from .plot import plot_ax_scalar
from .utils import create_dir


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Gaussian function """
    return amplitude * np.exp(-((x - x0) / sigma_x)**2
                              - ((y - y0) / sigma_y)**2)


def two_gaussians(x, y, amplitude, x0, y0, sigma_x, sigma_y, x01, y01, sigma_x1, sigma_y1):
    """ Two gaussians function """
    return amplitude * (np.exp(-((x - x0) / sigma_x)**2 - ((y - y0) / sigma_y)**2) + 
                        np.exp(-((x - x01) / sigma_x1)**2 - ((y - y01) / sigma_y1)**2))


def triangle(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Triangle function """
    return (amplitude * np.maximum(1 - abs((x - x0) / sigma_x), np.zeros_like(x)) 
                * np.maximum(1 - abs((y - y0) / sigma_y), np.zeros_like(x)))


def step(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    """ Step function """
    return (amplitude * np.where(abs(x - x0) / sigma_x < 0.5, np.ones_like(x), np.zeros_like(x))
            * np.where(abs(y - y0) / sigma_y < 0.5, np.ones_like(x), np.zeros_like(x)))


def sin2D(x, y, amplitude, Lx, Ly, n, m):
    """ 2D sines mode """
    return amplitude * np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly)


def gaussians(x, y, params):
    """ Multiple gaussians with multiple amplitude """
    profile = np.zeros_like(x)
    ngauss = int(len(params) / 5)
    params = np.array(params).reshape(ngauss, 5)
    for index in range(ngauss):
        profile += gaussian(x, y, *params[index, :])
    return profile


def random1D(x, amplitude, n_res, sigma=None):
    """ Generate random 2D profile using bicubic interpolation

    Args:
        x (numpy.ndarray): x coordinate
        amplitude (float): amplitude of the random profile
        n_res (int): number by which we divide the grid to get the random grid
        sigma (float, optional): if present then apply gaussian mask

    Returns:
        (numpy.ndarray): 1D random profile
    """
    xmin, xmax, npts = np.min(x), np.max(x), len(x)
    n_lower = int(npts / n_res)
    x_lower = np.linspace(xmin, xmax, n_lower)
    random_1D = amplitude * (2 * np.random.random(n_lower) - 1)
    f = interpolate.interp1d(x_lower, random_1D, kind='cubic')
    x = np.linspace(xmin, xmax, npts)
    if sigma is None:
        return f(x)
    else:
        x0 = (xmax + xmin) / 2
        return f(x) * np.exp(-((x - x0)**2 / sigma**2))


def random2D(X, Y, amplitude, n_res, sigma=None):
    """ Generate random 2D profile using bicubic interpolation

    Args:
        X (numpy.ndarray): X coordinate
        Y (numpy.ndarray): Y coordinate
        amplitude (float): amplitude of the random profile
        n_res (int): number by which we divide the grid to get the random grid
        sigma (float, optional): if present then apply gaussian mask
    Returns:
        (numpy.ndarray): 2D random profile
    """
    xmin, xmax, npts = np.min(X), np.max(X), len(X[0, :])
    n_lower = int(npts / n_res)
    x_lower, y_lower = np.linspace(xmin, xmax, n_lower), np.linspace(xmin, xmax, n_lower)
    z_lower = amplitude * (2 * np.random.random((n_lower, n_lower)) - 1)
    f = interpolate.interp2d(x_lower, y_lower, z_lower, kind='cubic')
    x, y = np.linspace(xmin, xmax, npts), np.linspace(xmin, xmax, npts)
    if sigma is None:
        return f(x, y)
    else:
        x0, y0 = (xmax + xmin) / 2, (xmax + xmin) / 2
        return f(x, y) * np.exp(-((X - x0)**2 / sigma**2 + (Y - y0)**2 / sigma**2))


def cosine_hill(x, y, amplitude, x0, y0, powx, powy, L):
    return amplitude * np.cos(np.pi / L * (x - x0)) ** powx * np.cos(np.pi / L * (y - y0)) ** powy


def parabol(x, y, L):
    return (1 - ((x - L / 2) / (L / 2)) ** 2) * (1 - ((y - L / 2) / (L / 2)) ** 2)


if __name__ == '__main__':
    fig_dir = 'figures/'
    create_dir(fig_dir)

    xmin, xmax, nnx = 0, 0.01, 101
    ymin, ymax, nny = 0, 0.01, 101
    x, y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
    X, Y = np.meshgrid(x, y)
    profiles = {
        'gaussian': gaussian(X, Y, 1.0, 0.5e-2, 0.5e-2, 1.0e-3, 1.0e-3),
        'off_gaussian': gaussian(X, Y, 1.0, 0.4e-2, 0.35e-2, 1.0e-3, 1.0e-3),
        'two_gaussians': two_gaussians(X, Y, 1.0, 0.4e-2, 0.5e-2, 1.0e-3, 1.0e-3, 0.6e-2, 0.5e-2, 1.0e-3, 1.0e-3),
        'random_4': random2D(X, Y, 1.0, 4, 3.0e-3),
        'random_8': random2D(X, Y, 1.0, 8, 3.0e-3),
        'random_16': random2D(X, Y, 1.0, 16, 3.0e-3),
    }

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
    for i, profile in enumerate(profiles.values()):
        plot_ax_scalar(fig, axes[i // 3][i % 3], X, Y, profile, '', geom='xy', cbar=False)
        axes[i // 3][i % 3].set_xticks([])
        axes[i // 3][i % 3].set_yticks([])
    fig.savefig(fig_dir + 'profiles', bbox_inches='tight')
