########################################################################################################################
#                                                                                                                      #
#                                         Drift-diffusion fluid plasma solver                                          #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import yaml
import copy
from cfdsolver import ScalarTransport

def gaussian(x, y, amplitude, x0, y0, sigma_x, sigma_y):
    return amplitude * np.exp(-((x - x0) / sigma_x) ** 2 - ((y - y0) / sigma_y) ** 2)

def main(config):
    """ Main function containing initialisation, temporal loop and outputs. Takes a config dict as input. """

    sim = ScalarTransport(config)

    # Gaussian initialization
    sim.u = gaussian(sim.X, sim.Y, 1, 0.5, 0.5, 1e-1, 1e-1)

    # Print header to sum up the parameters
    sim.print_init()

    # Iterations
    for it in range(1, sim.nit + 1):
        sim.dtsum += sim.dt

        # Calculation of diffusive flux
        sim.compute_dflux()

        # Update of the residual to zero
        sim.res[:] = 0

        # Loop on the cells to compute the interior flux and update residuals
        sim.compute_flux()

        # Impose boundary conditions
        sim.impose_bc()

        # Update residulas u -= res * dt / voln
        sim.update_res()

        # Post processing (printing and plotting)
        sim.postproc(it)


if __name__ == '__main__':

    with open('scalar.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)
