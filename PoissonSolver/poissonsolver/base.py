import numpy as np
import matplotlib.pyplot as plt
from poissonsolver.operators import grad, lapl

class BasePoisson:
    def __init__(self, xmin, xmax, nnx, ymin, ymax, nny):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.Lx, self.Ly = xmax - xmin, ymax - ymin
        self.dx, self.dy = (xmax - xmin) / (nnx - 1), (ymax - ymin) / (nny - 1)
        self.x, self.y = np.linspace(xmin, xmax, nnx), np.linspace(ymin, ymax, nny)
        self.nnx, self.nny = nnx, nny

        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Sum of the potentials
        self.potential = np.zeros_like(self.X)

        # Radius of the nodes for axisymmetric configuration
        self.R_nodes = None

    def compute_voln(self):
        """ Computes the nodal volume associated to each node (i, j) """
        voln = np.ones_like(self.X) * self.dx * self.dy
        voln[:, 0], voln[:, -1], voln[0, :], voln[-1, :] = \
            self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2, self.dx * self.dy / 2
        voln[0, 0], voln[-1, 0], voln[0, -1], voln[-1, -1] = \
            self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4, self.dx * self.dy / 4
        return voln

    @property
    def E_field(self):
        return - grad(self.potential, self.dx, self.dy, self.nnx, self.nny)
    
    @property
    def lapl(self):
        return lapl(self.potential, self.dx, self.dy, self.nnx, self.nny, r=self.R_nodes)

    @classmethod
    def round_up(cls, n, decimals=0):
        power = int(np.log10(n))
        digit = n / 10**power
        return np.ceil(digit) * 10**power
    
    @staticmethod
    def plot_ax_trial_1D(ax, x, function, n_points, title, direction='y', ylim=None):
        list_cut = [0, 0.25, 0.5, 0.75, 1.0]
        for cut_pos in list_cut:
            n = int(cut_pos * (n_points - 1))
            ax.plot(x, function[n, :], label='%s = %.2f %smax' % (direction, cut_pos, direction))
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        if ylim is not None:
            ax.set_ylim(ylim)
    
    @staticmethod
    def plot_ax_scalar(fig, ax, X, Y, field, title, colormap='RdBu', axi=False):
        if colormap == 'RdBu':
            # max_value = BasePoisson.round_up(np.max(np.abs(field)), decimals=2)
            max_value = np.max(np.abs(field))
            levels = np.linspace(- max_value, max_value, 101)
        else:
            levels = 101
        cs1 = ax.contourf(X, Y, field, levels, cmap=colormap)
        fraction_cbar = 0.1
        if axi: 
            ax.contourf(X, - Y, field, levels, cmap=colormap)
            aspect = 1.7 * np.max(Y) / fraction_cbar / np.max(X)
        else:
            aspect = 0.85 * np.max(Y) / fraction_cbar / np.max(X)
        fig.colorbar(cs1, ax=ax, pad=0.05, fraction=fraction_cbar, aspect=aspect)
        ax.set_aspect("equal")
        ax.set_title(title)

    @staticmethod
    def plot_ax_vector_arrow(fig, ax, X, Y, vector_field, name, colormap='Blues', axi=False):
        norm_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
        arrow_step = 20
        levels = np.linspace(0, np.max(norm_field), 101)
        CS = ax.contourf(X, Y, norm_field, levels, cmap=colormap)
        fraction_cbar = 0.1
        if axi:
            ax.contourf(X, - Y, norm_field, levels, cmap=colormap)
            aspect = 1.7 * np.max(Y) / fraction_cbar / np.max(X)
        else:
            aspect = 0.85 * np.max(Y) / fraction_cbar / np.max(X)
        fig.colorbar(CS, pad=0.05, fraction=fraction_cbar, ax=ax, aspect=aspect)
        ax.quiver(X[::arrow_step, ::arrow_step], Y[::arrow_step, ::arrow_step], 
                    vector_field[0, ::arrow_step, ::arrow_step], vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
        if axi:
            ax.quiver(X[::arrow_step, ::arrow_step], - Y[::arrow_step, ::arrow_step], 
                vector_field[0, ::arrow_step, ::arrow_step], - vector_field[1, ::arrow_step, ::arrow_step], pivot='mid')
        ax.set_title(name)
        ax.set_aspect('equal')

    def plot_2D(self, figname, axi=False):

        fig, axes = plt.subplots(ncols=3, figsize=(11, 14))

        self.plot_ax_scalar(fig, axes[0], self.X, self.Y, self.potential, 'Potential', axi=axi)

        E = self.E_field
        self.plot_ax_vector_arrow(fig, axes[1], self.X, self.Y, E, 'Electric field', axi=axi)

        lapl_field = self.lapl
        self.plot_ax_scalar(fig, axes[2], self.X, self.Y, - lapl_field, '- Laplacian', axi=axi)

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()

    def plot_1D2D(self, figname, axi=False):
        # 1D vector
        x = self.X[0, :]

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 14))

        self.plot_ax_scalar(fig, axes[0][0], self.X, self.Y, self.potential, 'Potential', axi=axi)
        self.plot_ax_trial_1D(axes[0][1], x, self.potential, self.nny, '1D cuts')

        E = self.E_field
        normE = np.sqrt(E[0]**2 + E[1]**2)
        self.plot_ax_vector_arrow(fig, axes[1][0], self.X, self.Y, E, 'Electric field', axi=axi)
        self.plot_ax_trial_1D(axes[1][1], x, normE, self.nny, '1D cuts', ylim=[0.99 * np.min(normE), 1.01 * np.max(normE)])

        lapl_field = self.lapl
        self.plot_ax_scalar(fig, axes[2, 0], self.X, self.Y, - lapl_field, '- Laplacian', axi=axi)
        self.plot_ax_trial_1D(axes[2][1], x, -  lapl_field, self.nny, '1D cuts')

        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(figname, bbox_inches='tight')
        plt.close()
