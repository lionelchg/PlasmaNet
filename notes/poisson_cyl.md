# Notes about Poisson discretization in cylindrical structured mesh for axis nodes

Starting from the Poisson equation in finite volume formulation at an axis node $i$:

$$
\begin{aligned}
\frac{1}{V_i}\int_{V_i} \nabla^2 \phi \, \md V &= \frac{1}{r_iA_i} \int r \nabla^2 \phi \, \md r \, \md x \\
 & = \frac{8}{\Delta x \Delta r^2} \int_{A_i} \nabla_{2D} \cdot (r \nabla \phi) \, \md A \\
 & = \frac{8}{\Delta x \Delta r^2} \int_{\partial A_i} r \nabla \phi \cdot \vb{n} \, \md l \\
 & = \frac{8}{\Delta x \Delta r^2} \frac{\Delta r}{2} \frac{\phi_i - \phi_j}{\Delta r} \Delta x = \frac{4}{\Delta r^2}(\phi_i - \phi_j)
\end{aligned}
$$

This explains the factor 4 from the discretization around the axis in `poissonsolver/linsystem.py` in cylindrical coordinates. On the last line only the axis terms have been considered the $x$ axis terms being the same as the cartesian geometry case.