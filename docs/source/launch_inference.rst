Launching a plasma simulation with a trained network
==========================================================

Once the network is trained, we will now launch a plasma oscillation simulation with a trained network.

Plasma oscillation introduction
--------------------------------

One of the fundamental properties of plasmas is to maintain electric charge neu- trality at a macroscopic
scale under equilibrium conditions. When this macroscopic charge neutrality is disturbed, large Coulomb
forces come into play and tend to restore the macroscopic charge neutrality.
Electrons and positive ions with charge e are considered. Ion motion is neglected since
its mass is way larger than that of the electrons.

.. math::
   \frac{\partial^2n_e}{\partial t^2}+\omega_p^2 n_e = 0 ~~~~ {where} ~~~~
   \omega_p = \sqrt{\frac{n_e e^2}{m_e\varepsilon_0}}
