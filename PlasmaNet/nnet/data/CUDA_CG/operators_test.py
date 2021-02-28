#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .operators import grad, div, lapl, L1_error, L2_error


def plot_fig_scalar(X, Y, field, name, fig_name):
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    fig, ax = plt.subplots(figsize=(10, 5))
    CS = ax.contourf(X, Y, field, 100)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=ax, aspect=5)
    ax.set_aspect("equal")

    plt.savefig('figures/operators/' + fig_name, bbox_inches='tight')


def plot_fig_vector(X, Y, field, name, fig_name):
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    CS = axes[0].contourf(X, Y, field[0], 100)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=axes[0], aspect=5)
    axes[0].set_aspect("equal")
    CS1 = axes[1].contourf(X, Y, field[1], 100)
    cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.05, ax=axes[1], aspect=5)
    axes[1].set_aspect("equal")

    plt.savefig('figures/operators/' + fig_name, bbox_inches='tight')


def print_error(computed, analytical, name):
    print('%s L1 error = %.2e - L2 error = %.2e' % (name, L1_error(computed, analytical), L2_error(computed, analytical)))


if __name__ == '__main__':
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1

    nx, ny = 101, 101

    dx, dy = (xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1)

    x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    # Gradient test

    scalar_field = X ** 3 + Y ** 3
    gradient = grad(scalar_field, dx, dy, nx, ny)
    gradient_th = np.zeros((2, ny, nx))
    gradient_th[0, :] = 2 * X
    gradient_th[1, :] = 2 * Y
    print_error(gradient, gradient_th, 'Gradient')
    plot_fig_vector(X, Y, gradient, "Gradient", "operators/gradient_test")

    # Divergence test

    vector_field = np.zeros((2, ny, nx))
    vector_field[0, :] = X ** 2
    vector_field[1, :] = Y ** 2
    # vector_field[0, :] = np.sin(X)
    # vector_field[1, :] = np.cos(Y)

    divergence_2 = div(vector_field, dx, dy, nx, ny)
    divergence_4 = div(vector_field, dx, dy, nx, ny, order=4)
    divergence_th = 2 * X + 2 * Y
    # divergence_th = np.cos(X) - np.sin(Y)

    print_error(divergence_2, divergence_th, 'Div 2nd order')
    print_error(divergence_4, divergence_th, 'Div 4th order')
    plot_fig_scalar(X, Y, divergence_2, "Divergence", "operators/divergence_test")

    # Laplacian test

    scalar_field = X ** 3 + Y ** 3

    laplacian_2 = lapl(scalar_field, dx, dy, nx, ny)
    laplacian_2_eq = div(gradient, dx, dy, nx, ny)
    laplacian_4 = lapl(scalar_field, dx, dy, nx, ny, order=4)
    laplacian_th = 6 * X + 6 * Y

    print_error(laplacian_2, laplacian_th, 'Lapl 2nd order')
    print_error(laplacian_2_eq, laplacian_th, 'Lapl from div grad 2nd order')
    print_error(laplacian_4, laplacian_th, 'Lapl 4th order')
    plot_fig_scalar(X, Y, laplacian_2, "Laplacian", "operators/laplacian_test")
    plot_fig_scalar(X, Y, laplacian_2_eq, "Laplacian", "operators/laplacian_test_eq")
