Launching a plasma simulation with a trained network
==========================================================

Once the network is trained, we will now launch a plasma oscillation simulation with a trained network.

Plasma oscillation introduction
--------------------------------

Plasmas tend to maintain electric charge neutrality at a macroscopic
scale at equilibrium. When this macroscopic charge neutrality is changed, Coulomb
forces try to restore the macroscopic charge neutrality. Ion motion is ignored since
they are way heavier than electrons. Thus, the electron density evolves harmanically.

.. math::
   \frac{\partial^2n_e}{\partial t^2}+\omega_p^2 n_e = 0 ~~~~ {where} ~~~~
   \omega_p = \sqrt{\frac{n_e e^2}{m_e\varepsilon_0}}

For further details about the plasma oscillation, please refer to [Cheng]_

Launching plasma simulations
-----------------------------
