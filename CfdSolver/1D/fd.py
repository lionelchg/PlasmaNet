#!/Users/cheng/code/envs/dl/bin/python
import sys
import os
import numpy as np
import copy
from numba import njit

from plot import plot_fd
from test_funcs import gaussian, step, packet_wave

fig_dir = 'figures/'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

schemes = ['FOU', 'LW', 'SOU']

@njit(cache=True)
def advance(res, u, sigma, scheme):
    if scheme == 'FOU':
        res[0] = sigma * (u[0] - u[-1])
        for i in range(1, len(u)):
            res[i] = sigma * (u[i] - u[i - 1])
    elif scheme == 'LW':
        res[0] = sigma / 2 * (u[1] - u[-2]) - sigma**2 / 2 * (u[1] + u[-2] - 2 * u[0])
        res[-1] = sigma / 2 * (u[1] - u[-2]) - sigma**2 / 2 * (u[1] + u[-2] - 2 * u[-1])
        for i in range(1, len(u) - 1):
            res[i] = (sigma / 2 * (u[i + 1] - u[i - 1]) 
                        - sigma**2 / 2 * (u[i + 1] + u[i - 1] - 2 * u[i]))
    elif scheme == 'SOU':
        res[0] = (sigma / 2 * (3 * u[0] - 4 * u[-2] + u[-3]) - sigma**2 / 2 * (u[-3] - 2 * u[-2] + u[0]))
        res[1] = (sigma / 2 * (3 * u[1] - 4 * u[0] + u[-2]) - sigma**2 / 2 * (u[-2] - 2 * u[0] + u[1]))
        for i in range(2, len(u)):
            res[i] = (sigma / 2 * (3 * u[i] - 4 * u[i - 1] + u[i - 2]) 
                        - sigma**2 / 2 * (u[i - 2] - 2 * u[i - 1] + u[i]))
        

@njit(cache=True)
def iterations(nt, res, u, sigma, scheme):
    # iterations
    for i in range(nt):
        advance(res, u, sigma, scheme)
        u -= res

def main():
    # Mesh properties
    xmin, xmax, nnx = 0, 2, 201
    Lx, ncx = xmax - xmin, nnx - 1
    x_th = np.linspace(xmin, xmax, 1001)
    dx = (xmax - xmin) / ncx

    # Number of schemes to test
    n_schemes = len(schemes)

    # CFL and simulation properties
    a = 1.0
    cfls = [0.25, 0.5, 0.8]

    for index, cfl in enumerate(cfls):
        dt = dx * cfl / a
        
        # initialization (number of timesteps required to do a full round)
        x = np.linspace(xmin, xmax, nnx)
        u_gauss = np.tile(gaussian(x, 1, 0.3), n_schemes).reshape(n_schemes,  nnx)
        u_step = np.tile(step(x, 1), n_schemes).reshape(n_schemes,  nnx)
        u_2pw = np.tile(packet_wave(x, 1, 0.5), n_schemes).reshape(n_schemes,  nnx)
        u_4pw = np.tile(packet_wave(x, 1, 0.25), n_schemes).reshape(n_schemes,  nnx)
        res = np.zeros_like(x)
        nt = int(2 * Lx / a / dt)

        print(f'CFL = {cfl:.2f} - nt = {nt:d}')

        # Iteration of the schemes
        for i_scheme, scheme in enumerate(schemes):
            iterations(nt, res, u_gauss[i_scheme, :], cfl, scheme)
            iterations(nt, res, u_step[i_scheme, :], cfl, scheme)
            iterations(nt, res, u_2pw[i_scheme, :], cfl, scheme)
            iterations(nt, res, u_4pw[i_scheme, :], cfl, scheme)
        plot_fd(x_th, x, u_gauss, u_step, u_2pw, u_4pw, 
                schemes, f'CFL = {cfl:.2f} - dx = {dx:.2e} m - dt = {dt:.2e} s - nits = {nt:d}', 
                fig_dir + f'cfl_{index}')

if __name__ == '__main__':
    main()