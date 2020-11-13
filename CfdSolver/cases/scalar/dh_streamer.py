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

from cfdsolver import StreamerMorrow

@profile
def main(config):
    """ Main function containing initialisation, temporal loop and outputs. Takes a config dict as input. """

    sim = StreamerMorrow(config)

    # Print information
    sim.print_init()

    nit, nny, nnx = sim.nit, sim.nny, sim.nnx

    if config['output']['dl_save'] == 'yes':
        potential_list = np.zeros((nit, nny, nnx))
        physical_rhs_list = np.zeros((nit, nny, nnx))
        if sim.photo:
            Sph_list = np.zeros((nit, nny, nnx))
            irate_list = np.zeros((nit, nny, nnx))

    # Iterations
    for it in range(1, sim.nit + 1):
        sim.dtsum += sim.dt

        # Solve poisson equation from charge distribution
        sim.solve_poisson()

        # Update of the residual to zero
        sim.resnd[:] = 0

        # Solve photoionization if activated
        if sim.photo and it % 10 == 1: sim.solve_photo()

        # Compute the chemistry source terms with or without photo
        sim.compute_chemistry(it)

        # Compute transport terms
        sim.compute_residuals()

        # Apply residuals
        sim.update_res()

        # Post processing of macro values
        sim.global_prop(it)

        # General post processing
        sim.postproc(it)

        if config['output']['dl_save'] == 'yes':
            potential_list[it - 1, :, :] = sim.potential + sim.backE * sim.X
            physical_rhs_list[it - 1, :, :] = sim.physical_rhs
            if sim.photo:
                irate_list[it - 1, :, :] = sim.irate
                Sph_list[it - 1, :, :] = sim.Sph

    sim.plot_global()
    if sim.save_data: np.save(sim.data_dir + 'globals', sim.gstreamer)

    if config['output']['dl_save'] == 'yes':
        np.save(config['output']['folder'] + config['casename'] + 'potential.npy', potential_list)
        np.save(config['output']['folder'] + config['casename'] + 'physical_rhs.npy', physical_rhs_list)
        if sim.photo:
            np.save(config['output']['folder'] + config['casename'] + 'Sph.npy', Sph_list)
            np.save(config['output']['folder'] + config['casename'] + 'irate.npy', irate_list)


if __name__ == '__main__':

    with open('dh_streamer.yml', 'r') as yaml_stream:
        cfg = yaml.safe_load(yaml_stream)

    main(cfg)
