Launching a plasma simulation with a trained network
==========================================================

Once the network is trained, we will now launch a plasma oscillation simulation with a trained network.

Plasma oscillation introduction
--------------------------------

Plasmas tend to maintain electric charge neutrality at a macroscopic
scale at equilibrium. When this macroscopic charge neutrality is changed, Coulomb
forces try to restore the macroscopic charge neutrality. Ion motion is ignored since
they are way heavier than electrons. Thus, the electron density evolves harmonically.

.. math::
   \frac{\partial^2n_e}{\partial t^2}+\omega_p^2 n_e = 0 ~~~~ {where} ~~~~
   \omega_p = \sqrt{\frac{n_e e^2}{m_e\varepsilon_0}}

For further details about the plasma oscillation, please refer to [Cheng]_

Launching plasma simulations
----------------------------

To launch a plasma simulation, we will first go to the folder ``path/to/plasmanet/CfdSolver/euler/poscill``


Reference with linear solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we will launch a plasma simulation with the linear solver. To do so, just modify the ``101.yml`` file.
As the linear solver will be used, we will only focus on the first block of the yaml file.
First the physical parameters of the simulation are defined:

.. code-block:: yaml

   casename: 'runs/reference_oscillation/'                  # Set the casename

   params:
      n_periods: 2.0                   # Number of oscillation periods we want to simulate
      geom: 'xy'
      dt: 1.0e-10                      # Fixed timestep
      nt_oscill: 5000

Then, we specify the initial profiles. We will start with a simple gaussian, but note that two two_gaussians
or sinusoidal profiles can be used as well. The ``init_func`` argument can either be a string, in this case, the ``init_args``
argument needs to be declared, or instead it can be a list containing a string and a list with the arguments:

.. code-block:: yaml

   init:
      n_back: 1.0e+16
      n_pert: 1.0e+11
      init_func: ['gaussian', [0.5e-2, 0.5e-2, 1.0e-3, 1.0e-3]]      # Second argument corresponds to the gaussian center

.. code-block:: yaml

   init:
      n_back: 1.0e+16
      n_pert: 1.0e+11
      func: 'two_gaussians'                           # Choose between 'gaussian', 'two_gaussians', 'sin2D', ...
      args: [0.4e-2, 0.5e-2, 1.0e-3, 1.0e-3,          # Coordinates of the center of the gaussians
                0.6e-2, 0.5e-2, 1.0e-3, 1.0e-3]

Concerning the Poisson equation and the simulation mesh, we will define the spatial discretization and the resolution type:

.. code-block:: yaml

   poisson:
      type: 'lin_system'            # Choose between network, lin_system, analytical and hybrid
      mat: 'cart_dirichlet'         # Plasma oscillation for now only in cartesian coordinates with D BC
      nmax_fourier: 10              # For the analytical solution

   mesh:
      xmin: 0
      ymin: 0
      xmax: 1.0e-2                  # Physical domain length in x direction
      ymax: 1.0e-2                  # Physical domain length in y direction
      nnx: 101                      # Number of points in x direction
      nny: 101                      # Number of points in y direction

   BC: 'full_out'                   # Problem boundary conditions

   output:
      save: 'plasma_period'         # Save for period intervals
      verbose: True                 # Useful information during inference
      period: 0.1                   # Saving every
      files: 'fig'                  # Saving options
      dl_save: 'no'                 # For training purposes
   globals:
      fig: [1.0, 1.5]               # Reference plotting points for final plot
      vars: 'yes'                   # Save global variables


Once the yaml file is configured according to your needs, just perform:

.. code-block:: shell

   plasma_euler -c 101.yml

Plasma oscillation with a CNN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Analogously, to launch a simulation with the network, just change:

.. code-block:: yaml

   poisson:
      type: 'network'               # Choose between network, lin_system, analytical and hybrid

And then we will focus on the remaining arguments of the config file. Please refer to the training section,
as the config file is quite similar, with just several small modifications. Just specify the network used for the
plasma simulation at the ``resume`` argument:

.. code-block:: yaml

   resume: '/path/to/trained/network/train/RF_study/Unet/5_scales/k_3/RF_200/models/random_8/model_best.pth'

If you don't have any trained networks, you can use the predefined configuration with the pre-trained network found at
``path/to/trained/network/model_best.pth``.
