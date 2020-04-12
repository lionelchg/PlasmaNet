import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['contour.negative_linestyle'] = 'solid'
default_cmap = 'RdBu'  # 'RdBu'

def round_up(n, decimals=0): 
    multiplier = 10 ** decimals 
    return np.ceil(n * multiplier) / multiplier

def plot_fig_list(X, Y, field_list, name_list, fig_name, colormap=default_cmap):
    n_fields = len(field_list)
    nfig_x, nfig_y = n_fields * 6, 6
    fig, axes = plt.subplots(ncols=n_fields, figsize=(nfig_x, nfig_y))
    for i, field in enumerate(field_list):
        max_value = round_up(np.max(np.abs(field)), decimals=2)
        levels = np.linspace(-max_value, max_value, 101)
        CS = axes[i].contourf(X, Y, field, levels, cmap=colormap)
        cbar = fig.colorbar(CS, pad=0.05, fraction=0.1, ax=axes[i], aspect=5)
        axes[i].set_aspect("equal")
        axes[i].set_title(name_list[i])

    fig.savefig('figures/' + fig_name, bbox_inches='tight')
    plt.close(fig)

def plot_fig_scalar(X, Y, field, name, fig_name, colormap=default_cmap):
    fig, ax = plt.subplots(figsize=(10, 5))
    CS = ax.contourf(X, Y, field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=ax, aspect=5)
    ax.set_aspect("equal")

    fig.savefig('figures/' + fig_name, bbox_inches='tight')
    plt.close(fig)


def plot_fig_vector(X, Y, field, name, fig_name, colormap=default_cmap):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    CS = axes[0].contourf(X, Y, field[0], 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=axes[0], aspect=5)
    axes[0].set_aspect("equal")
    CS1 = axes[1].contourf(X, Y, field[1], 100, cmap=colormap)
    cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.05, ax=axes[1], aspect=5)
    axes[1].set_aspect("equal")

    fig.savefig('figures/' + fig_name, bbox_inches='tight')
    plt.close(fig)

def plot_vector_arrow(X, Y, vector_field, name, fig_name, colormap='Blues'):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 5
    fig, ax = plt.subplots(figsize=(10, 10))
    CS = ax.contourf(X, Y, norm_field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=ax, aspect=5)
    q = ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
                 label='Quiver key, length = 10', labelpos='E')
    ax.set_title(name)
    plt.savefig('figures/' + fig_name, bbox_inches='tight')


def plot_fig(X, Y, potential, physical_rhs, name='potential_2D', nit=None, no_rhs=False, colormap=default_cmap, cbar_centered=True):
    # Plotting the potential
    levels = 101
    if no_rhs:
        fig, ax2 = plt.subplots(figsize=(9, 7))
    else:
        if cbar_centered: 
            max_value = round_up(np.max(np.abs(physical_rhs)), decimals=1)
            levels = np.linspace(-max_value, max_value, 101)
        fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(14, 7))
        CS1 = ax1.contourf(X, Y, physical_rhs, levels, cmap=colormap)
        cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.08, ax=ax1, aspect=5)
        cbar1.ax.set_ylabel(r'$\rho/\epsilon_0$ [V.m$^{-2}$]')
        ax1.set_aspect("equal")
    if cbar_centered: 
        max_value =  round_up(np.max(np.abs(potential)), decimals=1)
        levels = np.linspace(-max_value, max_value, 101)
    CS2 = ax2.contourf(X, Y, potential, levels, cmap=colormap)
    cbar2 = fig.colorbar(CS2, pad=0.05, fraction=0.08, ax=ax2, aspect=5)
    cbar2.ax.set_ylabel('Potential [V]')
    ax2.set_aspect("equal")

    if nit == None:
        fig.savefig('figures/' + name, bbox_inches='tight')
    else:
        fig.savefig('figures/{}{:06d}'.format(name, nit), bbox_inches='tight')
    plt.close(fig)


def plot_ax(fig, axes, X, Y, potential, physical_rhs, colormap=default_cmap, levels=100, npot=None):
    # Plotting the potential
    ax1 = axes[0]
    CS1 = ax1.contourf(X, Y, physical_rhs, levels, cmap=colormap)
    cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.08, ax=ax1, aspect=5)
    ax1.set_aspect("equal")
    ax1.set_title(r'$\rho / \epsilon_0$ [V.m$^{-2}$]')
    ax2 = axes[1]
    CS2 = ax2.contourf(X, Y, potential, 100, cmap=colormap)
    cbar2 = fig.colorbar(CS2, pad=0.05, fraction=0.08, ax=ax2, aspect=5)
    ax2.set_aspect("equal")
    if npot is not None:
        ax2.set_title('$\\phi_{%d}$ [V]' % npot)
    else:
        ax2.set_title('$\\phi$ [V]')

    return CS1.levels

def plot_ax_set_1D(axes, x, pot, E_field_norm, lapl_pot, n_points, M):
    # 1D plot
    n_middle = int(n_points / 2)

    axes[0].plot(x, - lapl_pot[n_middle, :], '.', label='M = %d' % M)
    axes[0].legend()
    axes[1].plot(x, pot[n_middle, :])
    axes[1].set_title(r'$\phi$')
    axes[2].plot(x, E_field_norm[n_middle, :])
    axes[2].set_title(r'$\mathbf{E}$')


def plot_set_1D(x, physical_rhs, pot, E_field_norm, lapl_pot, n_points, figtitle, figname):
    # 1D plot
    n_middle = int(n_points / 2)
    fig, axes = plt.subplots(ncols=3, figsize=(15, 7))

    axes[0].plot(x, - lapl_pot[n_middle, :], '.', label=r'$\nabla^2 \phi$')
    axes[0].plot(x, physical_rhs[n_middle, :], label=r'$\rho / \epsilon_0$')
    axes[0].legend()
    axes[1].plot(x, pot[n_middle, :])
    axes[1].set_title(r'$\phi$')
    axes[2].plot(x, E_field_norm[n_middle, :])
    axes[2].set_title(r'$\mathbf{E}$')

    plt.suptitle(figtitle, y=1)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')

def plot_set_2D(X, Y, physical_rhs, pot, E, figtitle, figname, no_rhs=False):
    """ Matplotlib plots. """
    if no_rhs:
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        plot_ax_scalar(fig, axes[0], X, Y, pot, r'$\phi$')
        plot_ax_vector_arrow(fig, axes[1], X, Y, E, r'$\mathbf{E}$')
    else:
        fig, axes = plt.subplots(ncols=3, figsize=(16, 5))
        plot_ax_scalar(fig, axes[0], X, Y, physical_rhs, r'$\rho / \epsilon_0$')
        plot_ax_scalar(fig, axes[1], X, Y, pot, r'$\phi$')
        plot_ax_vector_arrow(fig, axes[2], X, Y, E, r'$\mathbf{E}$')

    plt.suptitle(figtitle)
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight')


def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu'):
    if colormap == 'RdBu':
        max_value = round_up(np.max(np.abs(field)), decimals=1)
        levels = np.linspace(- max_value, max_value, 101)
    else:
        levels = 101
    cs1 = ax.contourf(X, Y, field, levels, cmap=colormap)
    fig.colorbar(cs1, ax=ax, pad=0.05, fraction=0.1, aspect=7)
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues'):
    norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
    arrow_step = 10
    CS = ax.contourf(X, Y, norm_field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.1, ax=ax, aspect=7)
    q = ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
    ax.set_title(name)
    ax.set_aspect('equal')
