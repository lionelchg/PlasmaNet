import os
import numpy as np
import yaml

from operators import grad
from boundary import outlet_x, outlet_y, perio_x, perio_y, full_perio
from scheme import edge_flux
from plot import plot_scalar
from metric import compute_voln


def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

if __name__ == '__main__':

    with open('config.yml', 'r') as yaml_stream:
        config = yaml.safe_load(yaml_stream)

    # Mesh properties
    nnx, nny = config['mesh']['nnx'], config['mesh']['nny']
    ncx, ncy = nnx - 1, nny - 1
    xmin, xmax = config['mesh']['xmin'], config['mesh']['xmax']
    ymin, ymax = config['mesh']['ymin'], config['mesh']['ymax']
    Lx, Ly = xmax - xmin, ymax - ymin
    dx = (xmax - xmin) / ncx
    dy = (ymax - ymin) / ncy
    x = np.linspace(xmin, xmax, nnx)
    y = np.linspace(ymin, ymax, nny)

    # Grid construction
    X, Y = np.meshgrid(x, y)
    voln = compute_voln(X, dx, dy)
    sij = np.array([dy / 2, dx / 2])

    # Boundary conditions
    BC = config['BC']

    # Convection speed
    a = np.zeros((2, nny, nnx))
    a[0, :, :] = config['transport']['convection_x']
    a[1, :, :] = config['transport']['convection_y']
    norm_a = np.sqrt(a[0, :, :]**2 + a[1, :, :]**2)
    max_speed = np.max(norm_a)

    # Diffusion coefficient
    D = np.zeros_like(X)
    D = config['transport']['diffusion']
    max_D = np.max(D)


    # Creation of the figures directory and numbering of outputs
    fig_dir = 'figures/' + config['casename'] 
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    number = 1
    save_type = config['output']['save']
    period = config['output']['period']

    # Timestep calculation through cfl and fourier
    cfl = 0.5
    fourier = 0.15
    dt = min(cfl * dx / max_speed, fourier * dx**2 / max_D)
    dtsum = 0

    # Number of iterations
    nit = 100


    # Scalar and Residual declaration
    u, res = np.zeros_like(X), np.zeros_like(X)

    # Gaussian initialization
    u = gaussian(X, Y, 1, 0.5, 0.5, 1e-1, 1e-1)

    # Print header to sum up the parameters
    print('Number of nodes: nnx = %d - nny = %d' %(nnx, nny))
    print('Bounding box: (%.1e, %.1e), (%.1e, %.1e)' % (xmin, ymin, xmax, ymax))
    print('Transport: a = %.2e - D = %.2e' % (max_speed, max_D))
    print('dx = %.2e - dy = %.2e CFL = %.2e - Fourier = %.2e - Timestep = %.2e' % (dx, dy, cfl, fourier, dt))
    print('------------------------------------')
    print('Start of simulation')
    print('------------------------------------')
    print('{:>10} {:>16} {:>17}'.format('Iteration', 'Timestep [s]', 'Total time [s]', width=14))

    # Iterations
    for it in range(1, nit + 1):
        dtsum += dt
        # Calculation of diffusive flux
        diff_flux = D * grad(u, dx, dy, nnx, nny)
        # Update of the residual to zero
        res[:] = 0
        # Loop on the cells to compute the interior flux and update residuals
        for i in np.arange(ncx):
            for j in np.arange(ncy):
                edge_flux(res, a, u, diff_flux, sij, i, j, i + 1, j, 0)
                edge_flux(res, a, u, diff_flux, sij, i, j + 1, i + 1, j + 1, 0)
                edge_flux(res, a, u, diff_flux, sij, i, j, i, j + 1, 1)
                edge_flux(res, a, u, diff_flux, sij, i + 1, j, i + 1, j + 1, 1)
        # Boundary conditions
        if BC == 'full_perio':
            full_perio(res)
        elif BC == 'perio_x':
            perio_x(res)
            outlet_y(res, a, u, diff_flux, dx, 0)
            outlet_y(res, a, u, diff_flux, dx, -1)
        elif BC == 'perio_y':
            perio_y(res)
            outlet_x(res, a, u, diff_flux, dy, 0)
            outlet_x(res, a, u, diff_flux, dy, -1) 
        elif BC == 'full_out':
            outlet_y(res, a, u, diff_flux, dx, 0)
            outlet_y(res, a, u, diff_flux, dx, -1)
            outlet_x(res, a, u, diff_flux, dy, 0)
            outlet_x(res, a, u, diff_flux, dy, -1)


        u = u - res * dt / voln

        if save_type == 'iteration':
            if it % period == 0 or it == nit:
                print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, dt, dtsum, width=14))
                plot_scalar(X, Y, u, res, dtsum, number, fig_dir)
                number += 1
        elif save_type == 'time':
            if np.abs(dtsum - number * period) < 0.1 * dt or it == nit:
                print('{:>10d} {:{width}.2e} {:{width}.2e}'.format(it, dt, dtsum, width=14))
                plot_scalar(X, Y, u, res, dtsum, number, fig_dir)
                number += 1