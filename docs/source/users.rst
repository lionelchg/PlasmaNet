User guide and library capabilities
====================================

The three modules :mod:`PlasmaNet.nnet`, :mod:`PlasmaNet.cfdsolver` and :mod:`PlasmaNet.poissonsolver`
each have a corresponding folder in the root of the directory where these libraries are
used for studies.

``CfdSolver/``
*******************

Running the studied fluid simulations of plasma oscillation and double headed streamer

``dl/``
--------------------

Neural networks runs of plasma oscillation and double headed streamers

``euler/``
---------------------

Convective vortex case for validating the Euler equations Lax-Wendroff scheme and plasma oscillation test case.

``perf/``
----------------------

Study of the performance of each option to solve the Poisson equation 

``scalar/``
----------------------

Scalar advection is validated on a simple square geometry. 

``NNet/``
********************

This repository allows to train neural networks from configuration files and post-process the training (plotting things such as metrics and losses).

``PoissonSolver/``
********************

This repository contains four directories:

``analytical/``
--------------------

Study of the exact solution of the 2D cartesian Dirichlet Poisson problem. The solution
relies on the exact Green funtion that is expanded in [Jackson]_.

``datasets/``
--------------------

Generation of datasets for the deep neural networks.

``losses/``
--------------------

Study of the losses in reduced dimensions.

``tests/``
-------------------

Unit tests of Poisson resolution.

.. [Jackson] Classical Electrodynamics, John David Jackson, 1999, John Wiley & Sons.