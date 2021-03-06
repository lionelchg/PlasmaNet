Overview of the library
=========================

Main modules
-------------------------

The library contains four main modules:

:mod:`PlasmaNet.nnet`
**************************

The neural network implementation is found in this module. The ``pytorch`` library has been used
based on a `pytorch-template <https://github.com/victoresque/pytorch-template/>`_ made by 
@victoresque on GitHub. Abstract classes are inherited for specific usage.

:mod:`PlasmaNet.cfdsolver`
***************************

This module allows to solve the 2D advection-diffusion and Euler equations with an object-oriented
paradigm. One class is defined for each specific set of equations that are needed.
All classes build upon the :class:`PlasmaNet.cfdsolver.base.basesim.BaseSim` class. 

:mod:`PlasmaNet.poissonsolver`
******************************

Library to solve the Poisson equation using either the neural network, an analytical solution or
a classical linear system solver. 

:mod:`PlasmaNet.common`
**************************

Module that holds methods that are common to all the ``PlasmaNet`` modules including plotting,
profiles.

Interacting with the library
-----------------------------

Yaml configuration files
*******************************

The use of yaml configuration files is the primary way of interacting directly with the library. Scripts calls from inside the library usually take
as argument a yaml configuration file ``.yml``. 

Entry points
*******************************

Entry points are used whenever a new functionnality is created inside the library to alleviate the calling of the function.