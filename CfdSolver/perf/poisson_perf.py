########################################################################################################################
#                                                                                                                      #
#                                   Poisson benchmark between linear and DL solver                                     #
#                                                                                                                      #
#                                     Guillaume Bogopolsky, CERFACS, 01.03.2021                                        #
#                                                                                                                      #
########################################################################################################################

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from PlasmaNet.cfdsolver import PlasmaEulerDL, PlasmaEuler


def run(config):
    """
    Main function containing initialization and performance measurements as a temporal loop.
    Takes a config dict as input.
    """

    dir_list = os.listdir(config["network"]["resume"])
    config["network"]["casename"] = config["plasma"]["casename"]
    config["network"]["arch"]["type"] = search_arch(os.path.join(config["network"]["resume"], dir_list[-1], "model_best.pth"))

    # Initialize DL solver
    sim_dl = PlasmaEulerDL(config["plasma"], config["network"], log_perf=True)
    # Extract logger for the linear solver
    logger = sim_dl.logger
    # Initialize linear solver
    sim = PlasmaEuler(config["plasma"], logger=logger)

    if sim.verbose:
        sim.print_init()

    # Temporal loop
    for it in range(1, sim.nit + 1):
        sim.it, sim_dl.it = it, it
        sim.dtsum += sim.dt
        sim_dl.dtsum += sim_dl.dt
        sim.time[it - 1] = sim.dtsum
        sim_dl.time[it - 1] = sim_dl.dtsum

        # Update residuals to zero



