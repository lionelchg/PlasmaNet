#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def L1_error(y, y_th, ds, S):
  return torch.sum(ds / S * abs(y - y_th))


def L2_error(y, y_th, ds, S):
  return torch.sqrt(ds / S * torch.sum((y - y_th)**2))


def Linf_error(y, y_th):
    return torch.max(torch.abs(y - y_th))


def errors(y, y_th, ds, S):
    return L1_error(y, y_th, ds, S), L2_error(y, y_th, ds, S), Linf_error(y, y_th)


def print_error(computed, analytical, ds, S, name):
    print('{} L1 error = {:.2e} - L2 error = {:.2e} - Linf error = {:.2e}'.format(name,
                                                                                  L1_error(computed, analytical, ds, S),
                                                                                  L2_error(computed, analytical, ds, S),
                                                                                  Linf_error(computed, analytical)))


def div(field, dx, dy, nx, ny, order=2):
    divergence = torch.zeros((ny, nx))

    if order == 2:
        divergence[1:-1, 1:-1] = (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / (2 * dx) + \
                                 (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / (2 * dy)
    elif order == 4:
        divergence[1:-1, 1:-1] = (field[0, 1:-1, 2:] - field[0, 1:-1, :-2]) / (2 * dx) + \
                                       (field[1, 2:, 1:-1] - field[1, :-2, 1:-1]) / (2 * dy)
        divergence[2:-2, 2:-2] = (- field[0, 2:-2, 4:] + 8 * field[0, 2:-2, 3:-1]
                                  - 8 * field[0, 2:-2, 1:-3] + field[0, 2:-2, :-4]) / (12 * dx) + \
                                 (- field[1, 4:, 2:-2] + 8 * field[1, 3:-1, 2:-2]
                                  - 8 * field[1, 1:-3, 2:-2] + field[1, :-4, 2:-2]) / (12 * dy)

    # array sides except corners (respectively upper, lower, left and right sides)
    divergence[0, 1:-1] = (field[0, 0, 2:] - field[0, 0, :-2]) / (2 * dx) + \
                          (4 * field[1, 1, 1:-1] - 3 * field[1, 0, 1:-1] - field[1, 2, 1:-1]) / (2 * dy)
    divergence[-1, 1:-1] = (field[0, -1, 2:] - field[0, -1, :-2]) / (2 * dx) + \
                           (3 * field[1, -1, 1:-1] - 4 * field[1, -2, 1:-1] + field[1, -3, 1:-1]) / (2 * dy)
    divergence[1:-1, 0] = (4 * field[0, 1:-1, 1] - 3 * field[0, 1:-1, 0] - field[0, 1:-1, 2]) / \
                          (2 * dx) + (field[1, 2:, 0] - field[1, :-2, 0]) / (2 * dy)
    divergence[1:-1, -1] = (3 * field[0, 1:-1, -1] - 4 * field[0, 1:-1, -2] + field[0, 1:-1, -3]) / \
                           (2 * dx) + (field[1, 2:, -1] - field[1, :-2, -1]) / (2 * dy)

    # corners (respectively upper left, upper right, lower left and lower right)
    divergence[0, 0] = (4 * field[0, 0, 1] - 3 * field[0, 0, 0] - field[0, 0, 2]) / (2 * dx) + \
                       (4 * field[0, 1, 0] - 3 * field[0, 0, 0] - field[0, 2, 0]) / (2 * dy)
    divergence[-1, 0] = (4 * field[0, -1, 1] - 3 * field[0, -1, 0] - field[0, -1, 2]) / (2 * dx) + \
                        (3 * field[1, -1, 0] - 4 * field[1, -2, 0] + field[1, -3, 0]) / (2 * dy)
    divergence[0, -1] = (3 * field[0, 0, -1] - 4 * field[0, 0, -2] + field[0, 0, -3]) / (2 * dx) +\
                        (4 * field[1, 1, -1] - 3 * field[1, 0, -1] - field[1, 2, -1]) / (2 * dy)
    divergence[-1, -1] = (3 * field[0, -1, -1] - 4 * field[0, -1, -2] + field[0, -1, -3]) / (2 * dx) + \
                         (3 * field[1, -1, -1] - 4 * field[1, -2, -1] + field[1, -3, -1]) / (2 * dy)
    return divergence


def lapl(field, dx, dy, nx, ny, order=2, b=0):
    laplacian = torch.zeros((ny, nx))

    laplacian[1:-1, 1:-1] = (field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2 + \
                            (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2
    if order == 2:
        laplacian[1:-1, 1:-1] = (1 - b) * ((field[2:, 1:-1] + field[:-2, 1:-1] - 2 * field[1:-1, 1:-1]) / dy**2 +
                            (field[1:-1, 2:] + field[1:-1, :-2] - 2 * field[1:-1, 1:-1]) / dx**2) + \
                            b * (field[2:, 2:] + field[2:, :-2] + field[:-2, :-2] + field[:-2, 2:] - 4 * field[1:-1, 1:-1]) \
                            / (2 * dx**2)
    elif order == 4:
        laplacian[2:-2, 2:-2] = (- field[4:, 2:-2] + 16 * field[3:-1, 2:-2] - 30 * field[2:-2, 2:-2] + 16 * field[1:-3, 2:-2]
                                 - field[:-4, 2:-2]) / (12 * dy**2) + \
                                (- field[2:-2, 4:] + 16 * field[2:-2, 3:-1] - 30 * field[2:-2, 2:-2] + 16 * field[2:-2, 1:-3]
                                 - field[2:-2, :-4]) / (12 * dx**2)
        laplacian[1, 1:-1] = ((field[2, 1:-1] + field[0, 1:-1] - 2 * field[1, 1:-1]) / dy**2 +
                             (field[1, 2:] + field[1, :-2] - 2 * field[1, 1:-1]) / dx**2)
        laplacian[-2, 1:-1] = ((field[-1, 1:-1] + field[-3, 1:-1] - 2 * field[-2, 1:-1]) / dy**2 +
                              (field[-2, 2:] + field[-2, :-2] - 2 * field[-2, 1:-1]) / dx**2)
        laplacian[1:-1, 1] = (field[2:, 1] + field[:-2, 1] - 2 * field[1:-1, 1]) / dy**2 + \
                             (field[1:-1, 2] + field[1:-1, 0] - 2 * field[1:-1, 1]) / dx**2
        laplacian[1:-1, -2] = (field[2:, -2] + field[:-2, -2] - 2 * field[1:-1, -2]) / dy**2 + \
                              (field[1:-1, -1] + field[1:-1, -3] - 2 * field[1:-1, -2]) / dx**2

    laplacian[0, 1:-1] = \
        (2 * field[0, 1:-1] - 5 * field[1, 1:-1] + 4 * field[2, 1:-1] - field[3, 1:-1]) / dy**2 + \
        (field[0, 2:] + field[0, :-2] - 2 * field[0, 1:-1]) / dx**2
    laplacian[-1, 1:-1] = \
        (2 * field[-1, 1:-1] - 5 * field[-2, 1:-1] + 4 * field[-3, 1:-1] - field[-4, 1:-1]) / dy**2 + \
        (field[-1, 2:] + field[-1, :-2] - 2 * field[-1, 1:-1]) / dx**2
    laplacian[1:-1, 0] = \
        (field[2:, 0] + field[:-2, 0] - 2 * field[1:-1, 0]) / dy**2 + \
        (2 * field[1:-1, 0] - 5 * field[1:-1, 1] + 4 * field[1:-1, 2] - field[1:-1, 3]) / dx**2
    laplacian[1:-1, -1] = \
        (field[2:, -1] + field[:-2, -1] - 2 * field[1:-1, -1]) / dy**2 + \
        (2 * field[1:-1, -1] - 5 * field[1:-1, -2] + 4 * field[1:-1, -3] - field[1:-1, -4]) / dx**2

    # corners (respectively upper left, upper right, lower left and lower right)
    laplacian[0, 0] = \
        (2 * field[0, 0] - 5 * field[1, 0] + 4 * field[2, 0] - field[3, 0]) / dy**2 + \
        (2 * field[0, 0] - 5 * field[0, 1] + 4 * field[0, 2] - field[0, 3]) / dx**2
    laplacian[0, -1] = \
        (2 * field[0, -1] - 5 * field[1, -1] + 4 * field[2, -1] - field[3, -1]) / dy**2 + \
        (2 * field[0, -1] - 5 * field[0, -2] + 4 * field[0, -3] - field[0, -4]) / dx**2
    laplacian[-1, 0] = \
        (2 * field[-1, 0] - 5 * field[-2, 0] + 4 * field[-3, 0] - field[-4, 0]) / dy**2 + \
        (2 * field[-1, 0] - 5 * field[-1, 1] + 4 * field[-1, 2] - field[-1, 3]) / dx**2
    laplacian[-1, -1] = \
        (2 * field[-1, -1] - 5 * field[-2, -1] + 4 * field[-3, -1] - field[-4, -1]) / dy**2 + \
        (2 * field[0, -1] - 5 * field[0, -2] + 4 * field[0, -3] - field[0, -4]) / dx**2

    return laplacian


def grad(field, dx, dy, nx, ny):
    gradient = torch.zeros((2, ny, nx))

    gradient[0, :, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2 * dx)
    gradient[0, :, 0] = (4 * field[:, 1] - 3 * field[:, 0] - field[:, 2]) / (2 * dx)
    gradient[0, :, -1] = - (4 * field[:, -2] - 3 * field[:, -1] - field[:, -3]) / (2 * dx)

    gradient[1, 1:-1, :] = (field[2:, :] - field[:-2, :]) / (2 * dy)
    gradient[1, 0, :] = (4 * field[1, :] - 3 * field[0, :] - field[2, :]) / (2 * dy)
    gradient[1, -1, :] = - (4 * field[-2, :] - 3 * field[-1, :] - field[-3, :]) / (2 * dy)

    return gradient


def scalar_rot(field, dx, dy, nx, ny):
    rotational = torch.zeros((ny, nx))

    # first compute dfield_y / dx
    rotational[:, 1:-1] = (field[1, :, 2:] - field[1, :, :-2]) / (2 * dx)
    rotational[:, 0] = (4 * field[1, :, 1] - 3 * field[1, :, 0] - field[1, :, 2]) / (2 * dx)
    rotational[:, -1] = (3 * field[1, :, -1] - 4 * field[1, :, -2] + field[1, :, -3]) / (2 * dx)

    # second compute dfield_x / dy
    rotational[1:-1, :] -= (field[0, 2:, :] - field[0, :-2, :]) / (2 * dy)
    rotational[0, :] -= (4 * field[0, 1, :] - 3 * field[0, 0, :] - field[0, 2, :]) / (2 * dy)
    rotational[-1, :] -= (3 * field[0, -1, :] - 4 * field[0, -2, :] + field[0, -3, :]) / (2 * dy)

    return rotational

def create_grid():
    """
    Initializes a square cartesian mesh for the operators tests.

    Returns
    -------
    nchannels : int
        Number of channels

    nx, ny : int
        Number of elements

    dx, dy : float
        Step size

    X, Y : torch.Tensor
        Tensor containing the cartesian coordinates of size (ny, nx)
    """

    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    nx, ny = 101, 101
    nchannels = 10
    dx, dy = (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1)
    x, y = torch.linspace(xmin, xmax, nx), torch.linspace(ymin, ymax, ny)
    Y, X = torch.meshgrid(y, x)  # Pay attention to the reversed order of the axes with torch.Tensor !

    return nchannels, nx, ny, dx, dy, X, Y
