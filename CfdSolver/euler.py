########################################################################################################################
#                                                                                                                      #
#                                          Euler equations related routines                                            #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 04.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
from numba import njit

@njit(cache=True)
def compute_timestep(cfl, dx, U, press, gamma):
    speed = np.sqrt(gamma * press / U[0]) + np.sqrt((U[1]**2 + U[2]**2) / U[0])
    dt = cfl * dx / np.max(speed[:])
    return dt

@njit(cache=True)
def compute_flux(U, gamma, r, F, press, Tgas):
    """ Compute the 2D flux of the Euler equations 
    as well as pressure and temperature """
    press = (gamma - 1) * (U[3] - (U[1]**2 + U[2]**2) / 2 / U[0])
    Tgas = press / U[0] / r
    # rhou - rhov
    F[0, 0] = U[1]
    F[0, 1] = U[2]
    # rho u^2 + p - rho u v
    F[1, 0] = U[1]**2 / U[0] + press
    F[1, 1] = U[1] * U[2] / U[0]
    # rho u^2 + p - rho u v
    F[2, 0] = U[1] * U[2] / U[0]
    F[2, 1] = U[2]**2 / U[0] + press
    # u(rho E + p) - v(rho E + p)
    F[3, 0] = U[1] / U[0] * (U[3] + press)
    F[3, 1] = U[2] / U[0] * (U[3] + press)

@njit(cache=True)
def compute_res(U, F, press, dt, volc, snc, ncx, ncy, gamma, ndim, nvert, res, res_c, U_c):
    """ Compute residuals for the Lax-Wendroff scheme in a cell-vertex fashion """
    dF_c = np.zeros((4, 2))
    for i in range(ncx):
        for j in range(ncy):
            U_c[:, j, i] = 0.25 * (U[:, j, i] + U[:, j + 1, i]
                                  + U[:, j, i + 1] + U[:, j + 1, i + 1])
            res_c[:, j, i] = - (F[:, 0, j, i] * snc[0, 0] +  F[:, 1, j, i] * snc[0, 1]
                    + F[:, 0, j, i + 1] * snc[1, 0] +  F[:, 1, j, i + 1] * snc[1, 1]
                    + F[:, 0, j + 1, i + 1] * snc[2, 0] +  F[:, 1, j + 1, i + 1] * snc[2, 1]
                    + F[:, 0, j + 1, i] * snc[3, 0] +  F[:, 1, j + 1, i] * snc[3, 1])

            # Add central term
            res[:, j, i] += res_c[:, j, i] / nvert
            res[:, j, i + 1] += res_c[:, j, i] / nvert
            res[:, j + 1, i + 1] += res_c[:, j, i] / nvert
            res[:, j + 1, i] += res_c[:, j, i] / nvert
    
    res_c *= dt / volc

    # Add second order diffusion term
    for i in range(ncx):
        for j in range(ncy):
            press_c = (gamma - 1) * (U_c[3, j, i] - 
                    (U[1, j, i]**2 + U[2, j, i]**2) / 2 / U[0, j, i])
            rho    = U_c[0, j, i]
            u      = U_c[1, j, i] / rho
            v      = U_c[2, j, i] / rho
            H      = U_c[3, j, i] / rho + press_c / rho
            beta   = gamma - 1
            alpha  = 0.5 * (u**2 + v**2)

            drho   = res_c[0, j, i]
            drhou  = res_c[1, j, i]
            drhov  = res_c[2, j, i]
            drhoE  = res_c[3, j, i]

            rhodu  = drhou - u*drho
            rhodv  = drhov - v*drho
            drhouu = u*drhou + u*rhodu
            drhovv = v*drhov + v*rhodv
            drhouv = u*drhov + v*rhodu

            dP     = beta * (drhoE - u*drhou -v*drhov) + beta * alpha * drho
        
            drhoH  = drhoE + dP
            drhoHu = H*rhodu + u*drhoH
            drhoHv = H*rhodv + v*drhoH

            dF_c[0, 0] = drhou
            dF_c[1, 0] = drhouu + dP
            dF_c[2, 0] = drhouv
            dF_c[3, 0] = drhoHu

            dF_c[0, 1] = drhov
            dF_c[1, 1] = drhouv
            dF_c[2, 1] = drhovv + dP
            dF_c[3, 1] = drhoHv

            res[:, j, i] -= 0.5 * ( dF_c[:, 0] * snc[0, 0] + dF_c[:, 1] * snc[0, 1] )
            res[:, j, i + 1] -= 0.5 * ( dF_c[:, 0] * snc[1, 0] + dF_c[:, 1] * snc[1, 1] )
            res[:, j + 1, i + 1] -= 0.5 * ( dF_c[:, 0] * snc[2, 0] + dF_c[:, 1] * snc[2, 1] )
            res[:, j + 1, i] -= 0.5 * ( dF_c[:, 0] * snc[3, 0] + dF_c[:, 1] * snc[3, 1] )