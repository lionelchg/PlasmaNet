.. PlasmaNet documentation master file, created by
   sphinx-quickstart on Mon Mar  1 15:48:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PlasmaNet Manual
========================

Welcome! This is the documentation for ``PlasmaNet``, last updated Sept 28, 2021.  ``PlasmaNet`` is a Python library to study the capability of neural networks to solve the Poisson equation coupled to plasma simulations. To install ``PlasmaNet`` run in the root of the repository:

.. code-block:: shell

   pip install -r requirements.txt
   pip install -e .

Some environment variables need to be defined for the library (these lines can be added to a ``.bashrc`` file):

.. code-block:: shell

   export ARCHS_DIR=path/to/plasmanet/NNet/archs
   export POISSON_DIR=path/to/plasmanet/PoissonSolver/linsystem

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   intro
   overview
   users
   developers
   modules
   tutorials
