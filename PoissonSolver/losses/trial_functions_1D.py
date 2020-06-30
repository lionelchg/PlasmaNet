import os
import numpy as np
import matplotlib.pyplot as plt

def triangle(x, L, p):
    return (1 - np.abs(2 * (x - L/2) / L))**p

def triangle_derivative(x, L, p):
    return - 2 * p / L * np.sign(x - L/2) * triangle(x, L, p - 1)

def bell(x, L, p):
    return (1 - np.abs((x - L/2) * 2 / L)**p)

def bell_derivative(x, L, p):
    return - 2 * p / L * np.sign(x - L/2) * np.abs((x - L/2) * 2 / L)**(p - 1)

def plot_trial_functions(x, L, p_range, trial, trial_d, trialname, figname):
    fig, axes = plt.subplots(ncols=2, figsize=(15, 7))
    for p in p_range:
        axes[0].plot(x, trial(x, Lx, p), label='p = {:.1f}'.format(p))
        axes[1].plot(x, trial_d(x, Lx, p), label='p = {:.1f}'.format(p))
    axes[0].legend()
    axes[1].legend()
    axes[0].grid(True)
    axes[1].grid(True)
    axes[0].set_title(trialname)
    axes[1].set_title(trialname + ' Derivatives')

    plt.savefig(fig_dir + figname, bbox_inches='tight')


fig_dir = 'figures/gaussian/1D_trialfunctions/'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

if __name__ == '__main__':
    xmin, xmax = 0, 1
    Lx = xmax - xmin
    nx = 201
    x = np.linspace(xmin, xmax, nx)
    p_range = [0.2, 0.5, 1, 2, 3]
    plot_trial_functions(x, Lx, p_range, triangle, triangle_derivative, 'Triangles', 'triangles_1D')
    p_range = [0.1, 0.2, 0.5, 1, 2, 3, 5, 6]
    plot_trial_functions(x, Lx, p_range, bell, bell_derivative, 'Bells', 'bells_1D')