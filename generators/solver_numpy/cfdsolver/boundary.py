########################################################################################################################
#                                                                                                                      #
#                                            Boundary conditions functions                                             #
#                                                                                                                      #
#                                          Lionel Cheng, CERFACS, 22.04.2020                                           #
#                                                                                                                      #
########################################################################################################################
import numpy as np
import copy


def outlet_y(res, a, u, diff_flux, dx, yb):
    """ Outlet boundary condition in the y direction """
    flux = dx / 2 * (0.75 * (a[1, yb, 1:] * u[yb, 1:] + diff_flux[1, yb, 1:]) + 
                     0.25 * (a[1, yb, :-1] * u[yb, :-1] + diff_flux[1, yb, :-1]))
    res[yb, 1:] += flux
    flux = dx / 2 * (0.25 * (a[1, yb, 1:] * u[yb, 1:] + diff_flux[1, yb, 1:]) + 
                     0.75 * (a[1, yb, :-1] * u[yb, :-1] + diff_flux[1, yb, :-1]))
    res[yb, :-1] += flux

def outlet_x(res, a, u, diff_flux, dy, xb):
    """ Outlet boundary condition in the x direction """
    flux = dy / 2 * (0.75 * (a[1, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) + 
                     0.25 * (a[1, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]))
    res[1:, xb] += flux
    flux = dy / 2 * (0.25 * (a[1, 1:, xb] * u[1:, xb] + diff_flux[1, 1:, xb]) + 
                     0.75 * (a[1, :-1, xb] * u[:-1, xb] + diff_flux[1, :-1, xb]))
    res[:-1, xb] += flux

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