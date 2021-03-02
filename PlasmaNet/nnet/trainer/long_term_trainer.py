########################################################################################################################
#                                                                                                                      #
#                                             Long Term Loss class                                                     #
#                                                                                                                      #
#                             Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 07.12.2020                                   #
#                                                                                                                      #
########################################################################################################################

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
import scipy.constants as co
import multiprocessing as mp
from ctypes import c_bool
from time import sleep, perf_counter
from more_itertools import grouper

from ...cfdsolver.euler.plasma import PlasmaEuler

class LongTermPlasmaEuler(PlasmaEuler):
    def __init__(self, config, data, output):
        super().__init__(config)

        self.alpha = 0.1
        self.ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / self.Lx**2 + 1 / self.Ly**2)
        self.scaling_factor = 1.0e+6

        sigma = config['params']['sigma']
        x0, y0, sigma_x, sigma_y = 5e-3, 5e-3, sigma, sigma

        # First de-normalize the data
        re_scale_data = data[0] / (self.ratio * self.scaling_factor)
        # Rewrite the flux declaration, as it does not correspond to the default
        # initial configuration, but rather the loaded init!
        n_electron = ((co.epsilon_0 * re_scale_data / co.e) + self.n_back)
        self.U[0] = self.m_e * n_electron

        # Similar procedure for the potential
        potential_rhs = output[0] / self.scaling_factor
        self.poisson.potential = potential_rhs

    def solve_poisson_dl(self, work_pipe):
        """ Solve poisson equation with the model undergoing training on the parent process. """
        self.physical_rhs = - (self.U[0] / self.m_e - self.n_back) * co.e / co.epsilon_0
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :]
                                              * self.ratio * self.scaling_factor).float()
        # Communicate with parent process
        work_pipe.send(physical_rhs_torch)
        potential_torch = work_pipe.recv()

        # Convert back to numpy
        potential_rhs = potential_torch.numpy()[0, 0] / self.scaling_factor

        self.poisson.potential = potential_rhs
        self.E_field = self.poisson.E_field

        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1: self.E_max = np.max(self.E_norm)

    def initialize_Efield(self, potential_rhs):
        """ As the E field is initialized withing solve_poisson_dl, and the poisson equation
        has already been solved, this method initializes the E field with the networks output. """

        potential_rhs = potential_rhs[0] / self.scaling_factor
        self.poisson.potential = potential_rhs
        self.E_field = self.poisson.E_field

        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1: self.E_max = np.max(self.E_norm)


def init_subprocesses(num_procs):
    """
    Initialize num_procs subprocesses and their control infrastructure.
    They will be used for the long term loss computation for each image with Cfdsolver, and offload the poisson
    computation to the current process for GPU inference with the model being trained.
    Returns:
    * procs: the list of Process objects
    * child_on: ctype switch
    * inference_status: list of ctype booleans for the inference status of each child
    * parent_ctl_conns: list of control Pipes parent connections
    * parent_work_conns: list of work Pipes parent connections
    """
    ctx = mp.get_context("spawn")  # Faster than "spawn" in our test, which is not needed

    # Initialize control pipes to manage worker state (sleep, work, return)
    parent_ctl_conns, child_ctl_conns = [], []
    # Initialize work pipes for data transfer
    parent_work_conns, child_work_conns = [], []
    for i in range(num_procs):
        # Control pipes
        parent, child = ctx.Pipe()
        parent_ctl_conns.append(parent)
        child_ctl_conns.append(child)
        # Work pipes
        parent, child = ctx.Pipe()
        parent_work_conns.append(parent)
        child_work_conns.append(child)

    # Create child loop control shared variable (basically an on/off switch)
    child_on = ctx.Value(c_bool, True)

    # Create inference shared variable to track child state (whether they have finished the temporal loop or not)
    inference_status = []
    for i in range(num_procs):
        inference_status.append(ctx.Value(c_bool, True))

    # Create the processes
    procs = []
    for i in range(num_procs):
        p = ctx.Process(
            target=child_worker,
            args=(
                child_on,
                inference_status[i],
                child_ctl_conns[i],
                child_work_conns[i],
            ),
            daemon=True,  # stops child processes if parent is stopped
        )
        procs.append(p)

    # Start processes
    for p in procs:
        p.start()

    return procs, child_on, inference_status, parent_ctl_conns, parent_work_conns


def child_worker(child_on, inference, ctl_pipe, work_pipe):
    """
    Child process worker. Contains a loop which checks the control pipe for the status (sleep, work, return).
    work: receives simulation parameters from work_pipe, executes the simulation and inference
    return: returns the simulation results through work_pipe
    sleep: nothing to do, awaiting next workload.
    """
    while child_on:
        # Check if a new status is available on the control pipe
        if ctl_pipe.poll():
            state = ctl_pipe.recv()
        else:
            state = "sleep"

        if state == "work":
            # Read work_pipe for initialisation
            output, data, config, its = work_pipe.recv()
            # Entering inference
            inference.value = True
            rhs = apply_cfdsolver(work_pipe, output, data, config, its)
            # done
            inference.value = False
            state = "sleep"

        elif state == "return":
            work_pipe.send(rhs)
            state = "sleep"

        else:
            sleep(0.01)


def propagate(config, output, data, model, its, inference_status, ctl_pipes, work_pipes):
    """
    Performs the simulations for the iterations needed by the lt loss. Returns the rhs
    after `its` iterations of Cfdsolver.
    Uses the subprocesses previously initialised for cfdsolver, and executes GPU inferences as they are needed.
    Basically the parent worker.
    """
    batch_size, num_procs = len(output), len(ctl_pipes)
    procs = list(range(num_procs))  # List of proc number
    rhs_out = []
    perf = perf_counter()

    # Split the batch in group of images of size the number of subprocesses
    for current_images in grouper(range(batch_size), num_procs, fillvalue=None):
        # Remove None elements from last group (which may be smaller than the other and filled with None)
        current_images = [cur for cur in current_images if cur is not None]

        # Initialize subprocesses
        # Send signal to children to enter "work" state
        for image, proc in zip(current_images, procs):
            # Update child state
            ctl_pipes[proc].send("work")
            # Send initialization data
            work_pipes[proc].send([output[image], data[image], config, its])

        # Execute cfdsolver simulations
        # Wait start of inference for any subprocess
        while not ctypes_any(inference_status):
            sleep(0.01)
        # Enter inference loop
        it_count = 0
        active_count = 0
        while ctypes_any(inference_status):  # as long as some child is inferring
            # Check if data is available in a Connection and deal with it
            # This should allow asynchronous task execution
            it_count += 1
            active_pipes = mp.connection.wait(work_pipes, timeout=0.0001)  # Lists the connections with pending work
            active_count += len(active_pipes)
            # Send all available data as a single batch
            if len(active_pipes) > 0:
                # Buffer in a list
                torch_rhs_buf = []
                for work_pipe in active_pipes:
                    torch_rhs_buf.append(work_pipe.recv())
                # Convert the list to a single torch Tensor
                torch_rhs = torch.cat(torch_rhs_buf, dim=0)
                torch_potential = model(torch_rhs.cuda())
                # Split the Tensor along the batch dimension
                torch_potential_buf = torch.split(torch_potential.detach().cpu(), 1, dim=0)
                # Send data back to subprocesses
                for i in range(len(active_pipes)):
                    active_pipes[i].send(torch_potential_buf[i])
        # print("Average active pipes {}".format(active_count / it_count))

        # Receive results from each child
        # Send signal to children to enter "return" state
        for _, proc in zip(current_images, procs):  # Only iterate on procs where we expect an output
            ctl_pipes[proc].send("return")
        for _, proc in zip(current_images, procs):
            rhs_out.append(work_pipes[proc].recv())

    # Aggregate the list of Tensors as a batch Tensor
    rhs_lt = np.concatenate(rhs_out, axis=0)

    perf = perf_counter() - perf
    # print("Propagate perf: {}".format(perf))

    return rhs_lt


def ctypes_any(ctypes_list):
    """ Implementation of the any() function on a list of ctypes.c_bool objects. """
    nb_true = 0
    for val in ctypes_list:
        nb_true += val.value
    return nb_true > 0


def apply_cfdsolver(work_pipe, output, data, config, its):
    """ Target function for the Processes. """
    sim = LongTermPlasmaEuler(config, data, output)

    # Initially don't solve the Poisson equation!
    # But initialize the E field
    sim.it = 0
    sim.initialize_Efield(output)

    sim.compute_flux_cold()
    # Compute euler fluxes (without pressure)
    # sim.compute_flux()
    sim.compute_flux_cold()

    # Compute residuals in cell-vertex method
    sim.compute_res()

    # Compute residuals from electromagnetic terms
    sim.compute_EM_source()

    # boundary conditions
    sim.impose_bc_euler()

    # Apply residual
    sim.update_res()

    # Retrieve center variables
    sim.temporal_variables(0)

    # Iterations
    for it in range(1, its + 1):
        sim.it = it
        sim.dtsum += sim.dt
        sim.time[it - 1] = sim.dtsum

        # Update of the residual to zero
        sim.res[:], sim.res_c[:] = 0, 0

        # Solve poisson equation
        sim.solve_poisson_dl(work_pipe)

        # Compute euler fluxes (without pressure)
        # sim.compute_flux()
        sim.compute_flux_cold()

        # Compute residuals in cell-vertex method
        sim.compute_res()

        # Compute residuals from electromagnetic terms
        sim.compute_EM_source()

        # boundary conditions
        sim.impose_bc_euler()

        # Apply residual
        sim.update_res()

        # Retrieve center variables
        sim.temporal_variables(it)

    physical_rhs = - (sim.U[0] / sim.m_e - sim.n_back) * co.e / co.epsilon_0
    rhs_lt = (physical_rhs[np.newaxis, np.newaxis, :, :] * sim.ratio * sim.scaling_factor)

    # return rhs_lt
    return rhs_lt
