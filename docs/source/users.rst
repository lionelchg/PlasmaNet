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

This repository allows to train neural networks from configuration files and post-process the training (plotting things such as metrics and losses). Two main architectures
are studied in ``PlasmaNet``: UNet and MSNet architectures. Sketches are showcased below:

.. figure:: figures/unet3_rf.eps
    :align: center
    :width: 700

    Sketch of UNet

.. figure:: figures/msnet3_rf.eps
    :align: center
    :width: 700
    
    Sketch of MSNet


``PoissonSolver/``
********************

This repository contains four directories:

``analytical/``
--------------------

Study of the exact solution of the 2D cartesian Dirichlet Poisson problem. The solution
relies on the exact Green funtion that is expanded in [Jackson]_.

``datasets/``
--------------------

Generation of datasets for the deep neural networks. The main datasets are ``random`` and ``fourier`` datasets explained in the article.

``linsystem/``
--------------------

Different profiles of right hand side and boundary conditions are considered in this repository and their solutions from linear system solvers
are plotted.

``network/``
--------------------

Neural networks are evaluated in this repository. They can be evaluated either on datasets or on specific profiles like the ones presented in ``linsystem/``.

``perfs/``
--------------------

Performance of the different options for solving the Poisson equation is monitored in this repository.

``tests/``
-------------------

Unit tests of Poisson resolution.

.. [Jackson] Classical Electrodynamics, John David Jackson, 1999, John Wiley & Sons.