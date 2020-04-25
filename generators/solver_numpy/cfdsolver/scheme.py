########################################################################################################################
#                                                                                                                      #
#                                               Scalar residual schemes                                                #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np


def edge_flux(res, a, u, diff_flux, sij, i1, j1, i2, j2, dim):
    """ Convection-diffusion flux. Implemented with 1st order upwind scheme for convection
    and second order centered scheme for diffusion """
    scalar_product = 0.5 * (a[dim, j1, i1] + a[dim, j2, i2]) * sij[dim]
    if scalar_product >= 0:
        flux = scalar_product * u[j1, i1]
    else:
        flux = scalar_product * u[j2, i2]
    flux -= 0.5 * (diff_flux[dim, j1, i1] + diff_flux[dim, j2, i2]) * sij[dim]
    res[j1, i1] += flux
    res[j2, i2] -= flux
