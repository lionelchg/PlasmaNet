import os
import numpy as np
import matplotlib.pyplot as plt
import copy

fig_dir = 'figures/'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

def compute_voln(X, dx, dy):
    voln = np.ones_like(X) * dx * dy
    voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
        dx * dy / 2, dx * dy / 2, dx * dy / 2, dx * dy / 2
    voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
        dx * dy / 4, dx * dy / 4, dx * dy / 4, dx * dy / 4
    return voln

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def plot_ax_scalar(fig, ax, X, Y, field, title):
    levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap='RdBu')
    fig.colorbar(cs1, ax=ax, pad=0.05, fraction=0.1, aspect=7)
    ax.set_aspect("equal")
    ax.set_title(title)


if __name__ == '__main__':

    # Mesh properties
    nnx, nny = 51, 51
    ncx, ncy = nnx - 1, nny - 1
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    Lx, Ly = xmax - xmin, ymax - ymin
    dx = (xmax - xmin) / ncx
    dy = (ymax - ymin) / ncy
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)

    # Grid construction
    X, Y = np.meshgrid(x, y)
    voln = compute_voln(X, dx, dy)

    # Convection speed and timestep
    a = np.zeros((nny, nnx, 2))
    a[:, :, 0] = 1
    norm_a = np.sqrt(a[:, :, 0]**2 + a[:, :, 1]**2)
    print(norm_a)
    max_speed = np.max(norm_a)
    print('max speed = %.2e' % max_speed)
    cfl = 0.5
    dt = cfl * dx / max_speed
    print('dx = %.2e - CFL = %.2e - Timestep = %.2e' % (dx, cfl, dt))

    # Number of iterations
    nit = 100

    # Scalar and Residual declaration
    u, res = np.zeros_like(X), np.zeros_like(X)

    # Gaussian initialization
    u = gaussian(X, Y, 1, 0.5, 0.5, 1e-1, 1e-1)

    for it in range(nit):
        if it % 10 == 0 or it == nit - 1:
            print('it = %d' % it)
        res[:] = 0
        # Loop on the cells
        for i in np.arange(ncx):
            for j in np.arange(ncy):
                scalar_product = a[j, i, 0] * dy / 2
                if scalar_product >= 0:
                    flux = scalar_product * u[j, i]
                else:
                    flux = scalar_product * u[j, i + 1]
                res[j, i] += flux
                res[j, i + 1] -= flux


                if scalar_product >= 0:
                    flux = scalar_product * u[j + 1, i]
                else:
                    flux = scalar_product * u[j + 1, i + 1]
                res[j + 1, i] += flux
                res[j + 1, i + 1] -= flux

                scalar_product = a[j, i, 1] * dx / 2
                if scalar_product >= 0:
                    flux = scalar_product * u[j, i]
                else:
                    flux = scalar_product * u[j + 1, i]
                res[j, i] += flux
                res[j + 1, i] -= flux


                if scalar_product >= 0:
                    flux = scalar_product * u[j, i + 1]
                else:
                    flux = scalar_product * u[j + 1, i + 1]
                res[j, i + 1] += flux
                res[j + 1, i + 1] -= flux

        # # Corner (only for full periodic)
        # res[0, 0] = res[0, 0] + res[0, -1] + res[-1, 0] + res[-1, -1]
        # res[0, -1], res[-1, 0], res[-1, -1] = res[0, 0], res[0, 0], res[0, 0]

        # # Periodic boundary conditions - Sides
        res[1:-1, 0] += res[1:-1, -1]
        res[1:-1, -1] = copy.deepcopy(res[1:-1, 0])
        # res[0, 1:-1] += res[-1, 1:-1]
        # res[-1, 1:-1] = copy.deepcopy(res[0, 1:-1])

        flux = dx / 2 * (a[0, 1:, 1] * u[0, 1:] + a[0, :-1, 1] * u[0, :-1]) / 2
        res[0, 1:] += flux
        res[0, :-1] += flux

        flux = dx / 2 * (a[-1, 1:, 1] * u[-1, 1:] + a[-1, :-1, 1] * u[-1, :-1]) / 2
        res[-1, 1:] += flux
        res[-1, :-1] += flux

        # flux = a[:, 0] * dy / 2 * (u[1:, 0] + u[:-1, 0]) / 2
        # res[1:, 0] += flux
        # res[:-1, 0] += flux
        
        # flux = a[:, 0] * dy / 2 * (u[1:, -1] + u[:-1, -1]) / 2
        # res[1:, -1] += flux
        # res[:-1, -1] += flux


        u = u - res * dt / voln

        if it % 10 == 0 or it == (nit - 1):
            fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
            plot_ax_scalar(fig, axes[0], X, Y, u, "Scalar")
            plot_ax_scalar(fig, axes[1], X, Y, res, "Residual")
            plt.tight_layout()
            plt.savefig(fig_dir + 'instant_%d' % it, bbox_inches='tight')
