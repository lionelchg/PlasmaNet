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
import scipy.constants as co
import yaml

from cfdsolver import PlasmaEuler


def main(config):
    """ Main function containing initialization, temporal loop and outputs. Takes a config dict as input. """

    sim = PlasmaEuler(config)

    # Print header to sum up the parameters
    if sim.verbose:
        sim.print_init()

    # Iterations
    for it in range(1, sim.nit + 1):
        sim.dtsum += sim.dt
        sim.time[it - 1] = sim.dtsum
        # Update of the residual to zero
        sim.res[:], sim.res_c[:] = 0, 0

        # Solve poisson equation
        sim.solve_poisson()

        # Compute euler fluxes
        sim.compute_flux()

        # Compute residuals in cell-vertex method
        sim.compute_res()

        # Compute residuals from electro-magnetic terms
        sim.compute_EM_source()

        # boundary conditions
        sim.impose_bc_euler()
        
        # Apply residual
        sim.update_res()

        # Post processing
        sim.postproc(it)

        # Retrieve center variables 
        sim.center_variables(it)

    # Plot temporals
    sim.plot_temporal()


if __name__ == '__main__':

    with open('plasma_oscillation.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)