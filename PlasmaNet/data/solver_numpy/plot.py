import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
default_cmap = 'RdBu'


def plot_fig_scalar(X, Y, field, name, fig_name, colormap=default_cmap):
    fig, ax = plt.subplots(figsize=(10, 5))
    CS = ax.contourf(X, Y, field, 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=ax, aspect=5)
    ax.set_aspect("equal")

    plt.savefig('figures/' + fig_name, bbox_inches='tight')


def plot_fig_vector(X, Y, field, name, fig_name, colormap=default_cmap):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    CS = axes[0].contourf(X, Y, field[0], 100, cmap=colormap)
    cbar = fig.colorbar(CS, pad=0.05, fraction=0.05, ax=axes[0], aspect=5)
    axes[0].set_aspect("equal")
    CS1 = axes[1].contourf(X, Y, field[1], 100, cmap=colormap)
    cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.05, ax=axes[1], aspect=5)
    axes[1].set_aspect("equal")

    plt.savefig('figures/' + fig_name, bbox_inches='tight')


def plot_fig(X, Y, potential, physical_rhs, name='potential_2D', nit=None, no_rhs=False, colormap=default_cmap):
    # Plotting the potential
    if no_rhs:
        fig, ax2 = plt.subplots(figsize=(7, 7))
    else:
        fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(14, 7))
        CS1 = ax1.contourf(X, Y, physical_rhs, 100, cmap=colormap)
        cbar1 = fig.colorbar(CS1, pad=0.05, fraction=0.08, ax=ax1, aspect=5)
        cbar1.ax.set_ylabel(r'$\rho/\epsilon_0$ [V.m$^{-2}$]')
        ax1.set_aspect("equal")
    CS2 = ax2.contourf(X, Y, potential, 100, cmap=colormap)
    cbar2 = fig.colorbar(CS2, pad=0.05, fraction=0.08, ax=ax2, aspect=5)
    cbar2.ax.set_ylabel('Potential [V]')
    ax2.set_aspect("equal")

    if nit == None:
        plt.savefig('figures/' + name, bbox_inches='tight')
    else:
        plt.savefig('figures/' + name + str(nit), bbox_inches='tight')


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
