Getting started with PlasmaNet
===============================

First create a virtual environment for ``PlasmaNet`` to avoid version issues and to have a clean start.
Please use at least Python >= 3.7 . Make sure that the loaded python version contains the ``sqlite`` library,
(for kraken load ``python/anaconda3.8``)


.. code-block:: shell

   python3 -m venv plasma_env


Once the environment is created, activate it and don't forget to upgrade pip:

.. code-block:: shell

   source /path/to/your/plasma_env/bin/activate
   pip install --upgrade pip


Then, clone the ``PlasmaNet`` repository from the `CERFACS public GitLab <https://gitlab.com/cerfacs/plasmanet>`_:

.. code-block:: shell

   git clone git@gitlab.com:cerfacs/plasmanet.git


At this point, you're all set to begin the journey!

Follow the instructions of the GitLab repo to install the packages. This will notably install ``PyTorch``, our
framework of choice and ``TensorBoard`` for the monitoring of training jobs (cf. the corresponding tutorial):

.. code-block:: shell

   pip install -r requirements.txt
   pip install -e .

You also need to define the following environmnet variables:


.. code-block:: shell

   export ARCHS_DIR=path/to/plasmanet/NNet/archs
   export POISSON_DIR=path/to/plasmanet/PoissonSolver/linsystem

Now, you can test your install by running the tests with PyTest in each test directory:


.. code-block:: shell

   cd tests/nnet/operators
   pytest
   cd ../../poissonsolver
   pytest
