########################################################################################################################
#                                                                                                                      #
#                                            Boundary conditions functions                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 05.11.2020                                           #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import copy


def outlet_y(res, a, u, diff_flux, dx, yb, r=None):
    """ Outlet boundary conditions in the y direction """
    if yb == 0:
        flux_sign = -1
    elif yb == -1:
        flux_sign = 1
    flux = flux_sign * dx / 2 * (0.75 * (a[1, yb, 1:] * u[yb, 1:] + diff_flux[1, yb, 1:]) + 
                     0.25 * (a[1, yb, :-1] * u[yb, :-1] + diff_flux[1, yb, :-1]))
    if r is not None:
        res[yb, 1:] += flux * r
    else:
        res[yb, 1:] += flux
    flux = flux_sign * dx / 2 * (0.25 * (a[1, yb, 1:] * u[yb, 1:] + diff_flux[1, yb, 1:]) + 
                     0.75 * (a[1, yb, :-1] * u[yb, :-1] + diff_flux[1, yb, :-1]))

    if r is not None:
        res[yb, :-1] += flux * r
    else:
        res[yb, :-1] += flux

    # res[yb, 1:-1] += dx * a[1, yb, 1:-1] * u[yb, 1:-1]
    # res[yb, 0] += dx / 2 * a[1, yb, 0] * u[yb, 0]
    # res[yb, -1] += dx / 2 * a[1, yb, -1] * u[yb, -1]


def outlet_x(res, a, u, diff_flux, dy, xb, r=None):
    """ Outlet boundary condition in the x direction """
    if xb == 0:
        flux_sign = -1
    elif xb == -1:
        flux_sign = 1

    if r is not None:
        flux = flux_sign * dy / 2 * (0.75 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) * r[1:, xb] + 
                         0.25 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]) * r[:-1, xb])
        res[1:, xb] += flux
        flux = flux_sign * dy / 2 * (0.25 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) * r[1:, xb] + 
                         0.75 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]) * r[:-1, xb])
        res[:-1, xb] += flux
    else:
        flux = flux_sign * dy / 2 * (0.75 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) + 
                         0.25 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]))
        res[1:, xb] += flux
        flux = flux_sign * dy / 2 * (0.25 * (a[0, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) + 
                         0.75 * (a[0, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]))
        res[:-1, xb] += flux
    # res[1:-1, xb] += dy * a[0, 1:-1, xb] * u[1:-1, xb]
    # res[0, xb] += dy / 2 * a[0, 0, xb] * u[0, xb]
    # res[-1, xb] += dy / 2 * a[0, -1, xb] * u[-1, xb]


def perio_x(res):
    """ Periodic condition in the x direction """
    res[1:-1, 0] += res[1:-1, -1]
    res[1:-1, -1] = copy.deepcopy(res[1:-1, 0])


def perio_y(res):
    """ Periodic condition in the y direction """
    res[0, 1:-1] += res[-1, 1:-1]
    res[-1, 1:-1] = copy.deepcopy(res[0, 1:-1])


def full_perio(res):
    """ Full periodic conditions in the 4 directions """
    # Corner (only for full periodic)
    res[0, 0] = res[0, 0] + res[0, -1] + res[-1, 0] + res[-1, -1]
    res[0, -1], res[-1, 0], res[-1, -1] = res[0, 0], res[0, 0], res[0, 0]

    # Periodic boundary conditions - Sides
    res[1:-1, 0] += res[1:-1, -1]
    res[1:-1, -1] = copy.deepcopy(res[1:-1, 0])
    res[0, 1:-1] += res[-1, 1:-1]
    res[-1, 1:-1] = copy.deepcopy(res[0, 1:-1])


def impose_bc(BC, res, a, u, diff_flux, dx, dy, geom, Y):
    """ Impose boundary conditions specified in the config file """
    # Boundary conditions
    if BC == 'full_perio':
        full_perio(res)
    elif BC == 'perio_x':
        perio_x(res)
        outlet_y(res, a, u, diff_flux, dx, 0)
        outlet_y(res, a, u, diff_flux, dx, -1)
    elif BC == 'perio_y':
        perio_y(res)
        outlet_x(res, a, u, diff_flux, dy, 0)
        outlet_x(res, a, u, diff_flux, dy, -1)
    elif BC == 'full_out':
        if geom == 'xy':
            outlet_y(res, a, u, diff_flux, dx, 0)
            outlet_y(res, a, u, diff_flux, dx, -1)
            outlet_x(res, a, u, diff_flux, dy, 0)
            outlet_x(res, a, u, diff_flux, dy, -1)
        elif geom == 'xr':
            outlet_y(res, a, u, diff_flux, dx, -1, r=np.max(Y))
            outlet_x(res, a, u, diff_flux, dy, 0, r=Y)
            outlet_x(res, a, u, diff_flux, dy, -1, r=Y)


def impose_bc_euler(BC, res):
    """ Full periodic conditions in the 4 directions """
    # Corner (only for full periodic)
    res[:, 0, 0] = res[:, 0, 0] + res[:, 0, -1] + res[:, -1, 0] + res[:, -1, -1]
    res[:, 0, -1], res[:, -1, 0], res[:, -1, -1] = res[:, 0, 0], res[:, 0, 0], res[:, 0, 0]

    # Periodic boundary conditions - Sides
    res[:, 1:-1, 0] += res[:, 1:-1, -1]
    res[:, 1:-1, -1] = copy.deepcopy(res[:, 1:-1, 0])
    res[:, 0, 1:-1] += res[:, -1, 1:-1]
    res[:, -1, 1:-1] = copy.deepcopy(res[:, 0, 1:-1])
