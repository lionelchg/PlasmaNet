########################################################################################################################
#                                                                                                                      #
#                               Convective vortex for validation of Euler integration                                  #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import yaml

from cfdsolver import Euler


def covo(x, y, x0, y0, u0, v0, rho0, p0, T0, alpha, K, gamma, r, t, U):
    """ Initialize isentropic convective vortex as given by idolikecfd chap 7. """
    xbar = x - x0 - u0 * t
    ybar = y - y0 - v0 * t
    rbar = np.sqrt(xbar**2 + ybar**2)
    u = u0 - K / (2 * np.pi) * ybar * np.exp(alpha * (1 - rbar**2) / 2)
    v = v0 + K / (2 * np.pi) * xbar * np.exp(alpha * (1 - rbar**2) / 2)
    T = T0 - K**2 * (gamma - 1) / (8 * alpha * np.pi**2 * gamma * r) * np.exp(alpha * (1 - rbar**2))
    rho = rho0 * (T / T0)**(1 / (gamma - 1))
    p = p0 * (T / T0)**(gamma / (gamma - 1))
    # Define conservative variables
    U[0] = rho                                          # density
    U[1] = rho * u                                      # momentum along x
    U[2] = rho * v                                      # momentum along y
    U[3] = rho / 2 * (u**2 + v**2) + p / (gamma - 1)    # total energy with closure on internal energy


def main(config):
    """ Main function containing initialization, temporal loop and outputs. Takes a config dict as input. """

    sim = Euler(config)

    # Convective vortex parameters and initialization
    x0, y0 = 0, 0
    u0, v0 = 2, 0
    rho0, p0 = 1, 1 / sim.gamma
    alpha, K = 1, 5
    a0 = np.sqrt(sim.gamma * p0 / rho0)
    T0 = 1 / sim.gamma / sim.r
    covo(sim.X, sim.Y, x0, y0, u0, v0, rho0, p0, T0, alpha, K, sim.gamma, sim.r, 0, sim.U)

    # Print header to sum up the parameters
    if sim.verbose:
        sim.print_init()

    # Iterations
    for it in range(1, sim.nit + 1):
        # Update of the residual to zero
        sim.res[:], sim.res_c[:] = 0, 0

        # Compute euler fluxes
        sim.compute_flux()

        # Calculate timestep based on the maximum value of u + ss
        sim.compute_timestep()

        # Compute residuals in cell-vertex method
        sim.compute_res()

        # boundary conditions
        sim.impose_bc_euler()
        
        # Apply residual
        sim.update_res()

        # Post processing
        sim.postproc(it)


if __name__ == '__main__':

    with open('covo.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)
