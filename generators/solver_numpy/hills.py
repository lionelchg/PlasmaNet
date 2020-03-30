import numpy as np
import matplotlib.pyplot as plt

# Global variables
n_points = 64
xmin, xmax = 0, 0.01
dx = (xmax - xmin) / (n_points - 1)
x = np.linspace(xmin, xmax, n_points)

L = xmax - xmin

ni0 = 1e16
sigma_min, sigma_max = 1e-3, 3e-3
pow_min, pow_max = 3, 7
x0 = 5e-3


def gaussian(x, amplitude, x0, sigma_x):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2)


def cosine_hill(x, amplitude, x0, powx, L):
    return amplitude * np.cos(np.pi / L * (x - x0)) ** powx


def parabol(x, L):
    return 1 - ((x - L / 2) / (L / 2)) ** 2


if __name__ == '__main__':
    fig, axes = plt.subplots(ncols=2, figsize=(13, 6))

    sigma_range = np.linspace(sigma_min, sigma_max, 5)
    pow_range = np.linspace(pow_min, pow_max, 3)
    x0_range = np.linspace(3e-3, 7e-3, 3)
    for sigma in sigma_range:
        axes[0].plot(x, gaussian(x, ni0, x0, sigma), label='Sigma = %.1e' % sigma)
    axes[0].legend()
    for powx in pow_range:
        for x_c in x0_range:
            axes[1].plot(x, cosine_hill(x, ni0, x_c, powx, L),
                         label='Pow = %.1e x_c = %.1e' % (powx, x_c))
    # axes[1].legend()
    plt.savefig('FIGURES/HILLS/hills_1D', bbox_inches='tight')
