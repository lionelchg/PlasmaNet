import numpy as np
import matplotlib.pyplot as plt

from poissonsolver.operators import lapl, grad, derivative
from poissonsolver.plot import plot_set_1D, plot_set_2D, plot_ax_set_1D, plot_ax_scalar, plot_ax_vector_arrow
from poissonsolver.linsystem import laplace_square_matrix, dirichlet_bc
from poissonsolver.postproc import lapl_diff, compute_voln, func_energy, func_energy_torch
import poissonsolver.operators_torch as optorch

print_bool = False

def compute_values(ampl_potential, trial_function, print_bool=False):
    potential = ampl_potential * trial_function
    E_field = - grad(potential, dx, dy, n_points, n_points)
    E_field_norm = np.sqrt(E_field[0]**2 + E_field[1]**2)
    functional_energy = func_energy(potential, E_field, physical_rhs, voln) - functional_energy_target
    points_loss = np.sum((potential - potential_target)**2) / n_points**2
    lapl_pot = lapl(potential, dx, dy, n_points, n_points)
    interior_diff = lapl_diff(potential, physical_rhs, dx, dy, n_points, n_points)
    lapl_loss = np.sum(interior_diff**2) / n_points**2
    elec_loss = np.sum((E_field_norm - E_field_norm_target)**2) / n_points**2

    info_test = 'A = %.2e, energy = %.4e - points_loss = %.4e - lapl_loss = %.4e - elec_loss = %.4e' \
            % (ampl_potential, functional_energy, points_loss, lapl_loss, elec_loss)
    if print_bool: print(info_test)

    return potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, lapl_loss, elec_loss, info_test

def losses_1D(trial_function, amplmin, amplmax, nampl, figname):

    coef_range = np.linspace(amplmin, amplmax, nampl)
    list_loss = []
    for ampl_potential in coef_range:
        potential, E_field, E_field_norm, lapl_pot, functional_energy, points_loss, \
            lapl_loss, elec_loss, info_test = compute_values(ampl_potential, trial_function)
        list_loss.append([functional_energy, points_loss, lapl_loss, elec_loss])
    losses_1D = np.array(list_loss)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes[0][0].plot(coef_range, losses_1D[:, 0])
    axes[0][0].set_title('Energy')
    axes[0][0].grid()
    axes[0][1].plot(coef_range, losses_1D[:, 1])
    axes[0][1].set_title('Points Loss')
    axes[0][1].grid()
    axes[1][0].plot(coef_range, losses_1D[:, 2])
    axes[1][0].set_title('Lapl Loss')
    axes[1][0].grid()
    axes[1][1].plot(coef_range, losses_1D[:, 3])
    axes[1][1].set_title('Electric Loss')
    axes[1][1].grid()
    plt.tight_layout()
    plt.savefig(fig_dir + figname, bbox_inches='tight')

def plot_trial_function(X, Y, dx, dy, trial_function, n_points, figname):
    # 1D vector
    x = X[0, :]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 14))

    plot_ax_scalar(fig, axes[0][0], X, Y, trial_function, 'Trial function')
    plot_ax_trial_1D(axes[0][1], x, trial_function, n_points, '1D cuts')

    gradtrial = grad(trial_function, dx, dy, n_points, n_points)
    normgrad = np.sqrt(gradtrial[0]**2 + gradtrial[1]**2)
    plot_ax_vector_arrow(fig, axes[1][0], X, Y, gradtrial, 'Gradient')
    plot_ax_trial_1D(axes[1][1], x, normgrad, n_points, '1D cuts')

    lapl_trial = lapl(trial_function, dx, dy, n_points, n_points)
    plot_ax_scalar(fig, axes[2, 0], X, Y, - lapl_trial, '- Laplacian')
    plot_ax_trial_1D(axes[2][1], x, -  lapl_trial, n_points, '1D cuts')

    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')

def plot_ax_trial_1D(ax, x, function, n_points, title):
    direction = 'y'
    list_cut = [0, 0.25, 0.5]
    for cut_pos in list_cut:
        n = int(cut_pos * (n_points - 1))
        ax.plot(x, function[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
    ax.set_title(title)
    ax.legend()
    ax.grid(True)