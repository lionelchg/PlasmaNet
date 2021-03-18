import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_ticks(ax, labels_x, labels_y):
    """Useful functions to declare figure axes

    Args:
        ax (plt.ax): Axes on which ticks are plotted
        labels_x (list): List of strings with names
        labels_y (list): List of strings with names
    """
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(labels_y)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels_x)
    ax.set_yticklabels(labels_y)
    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #        rotation_mode="anchor")

class PlotRes:
    """ Class that contains three plots performed for Resolution plots
    """
    def __init__(self, residuals, dsets_type, dsets_res, loss_titles, fig_dir, max_val, min_val):
        """ Plots the MSE of all test cases, the average of gaussian and random test cases
        and an the global mean.

        Args:
            residuals (torch.tensor): torch tensor containing all the losses to plot
            dsets_type (list):  list of strings containing  the names of all the test cases evaluated
            dsets_res (list): list of integers containing all the evaluated resolutions
            loss_titles (list): list of strings containing the name of the evaluated metrics
            fig_dir (string): directory on which the plots are saved
            max_val (float): max value of tensors for plotting
            min_val (float): min value of tensors for plotting
        """

        self.data = residuals
        self.cases = dsets_type
        self.resolutions = dsets_res
        self.losses = loss_titles
        self.dir = fig_dir
        self.max_values = max_val
        self.min_values = min_val
    
    def generate_four(self, fig):
        """ Returns 4 axes for the 4 subplots of the image

        Args:
            fig (plt.fig): Figure to be divided in 4 subplots
        """

        gs = GridSpec(2, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[1, 0]) 
        ax2 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])

        return [ax1, ax2, ax3, ax4]
    
    def define_log(self, log):
        """ Helpful method for plotting log_scale or not

        Args:
            log (bool): If true, Imshow in Log Scale
        """
        if log:
            norm_plot = LogNorm()
        else:
            norm_plot = None
        return norm_plot

    
    def plot_all(self, fig_size, log):

        fig = plt.figure(constrained_layout=True, figsize=fig_size)
        axes_all = self.generate_four(fig)

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        cmap_e = plt.get_cmap('Blues')
        cmap_e.set_bad(color = 'k', alpha = 1.)

        for i, ax in enumerate(axes_all):
            # Initial time step masked
            if i < 2:
                im = ax.imshow(self.data[:, :, i], vmin=self.min_values[i], 
                    vmax=self.max_values[i], norm=self.define_log(log), cmap=cmap)
            else:
                im = ax.imshow(self.data[:, :, i], vmin=self.min_values[i], 
                    vmax=self.max_values[i], norm=self.define_log(log), cmap=cmap_e)

            # Create colorbars 
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            cbar1 = ax.figure.colorbar(im, ax=ax, cax=cax)

            # Only put spatial resolution ticks on the first column
            if i == 2 or i == 3:
                ax.set_xlabel('Spatial Resolution')
            if i == 0 or i == 2: 
                plot_ticks(ax, self.resolutions, self.cases)
            else:
                plot_ticks(ax, self.resolutions, [])
            ax.set_title('{}'.format(self.losses[i]))

        # Save figure
        fig.savefig(self.dir / 'MSE_all_residuals.png', bbox_inches='tight')
        plt.close()

    def plot_categories(self, test_cases_reduced, fig_size, log):

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        cmap_e = plt.get_cmap('Blues')
        cmap_e.set_bad(color = 'k', alpha = 1.)

        # Initialize grids, labels and colormaps
        self.test_cases_reduced = ['Gaussians', 'Random']
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

        axes_all = self.generate_four(fig)

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        # Create and store in new tensor
        losses_section = torch.zeros((2, len(self.resolutions), 5))
        losses_section[0, :, :] = torch.mean(self.data[:3, :], 0)
        losses_section[1, :, :] = torch.mean(self.data[3:, :], 0)
        self.losses_section = losses_section

        for i, ax in enumerate(axes_all):
            # Initial time step masked
            if i < 2:
                im = ax.imshow(losses_section[:, :, i+1], vmin=self.min_values[i + 1], 
                    vmax=self.max_values[i + 1], norm=self.define_log(log), cmap=cmap)
            else:
                im = ax.imshow(losses_section[:, :, i+1], vmin=self.min_values[i + 1], 
                    vmax=self.max_values[i + 1], norm=self.define_log(log), cmap=cmap_e)

            # Create colorbars and orientate ticks
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10%", pad=0.05)
            cbar1 = ax.figure.colorbar(im, ax=ax, cax=cax)

            plot_ticks(ax, self.resolutions, self.test_cases_reduced)

            # Only put spatial resolution ticks on the first column
            if i == 2 or i == 3:
                ax.set_xlabel('Spatial Resolution')

            if i == 1 or i == 3:
                plot_ticks(ax, self.resolutions, [])
            else:
                plot_ticks(ax, self.resolutions, self.test_cases_reduced)

            ax.set_title('{}'.format(self.losses[i]))

        fig.savefig(self.dir / 'MSE_category_losses.png', bbox_inches='tight')
        plt.close()

    def plot_overall_mean(self, fig_size, log):
        # Single Column Plot Mean of Gaussians and Randoms
        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        cmap_e = plt.get_cmap('Blues')
        cmap_e.set_bad(color = 'k', alpha = 1.)

        # Initialize Figure and colormaps
        fig = plt.figure(constrained_layout=True, figsize=fig_size)

        axes_all = self.generate_four(fig)

        cmap = plt.get_cmap('Reds')
        cmap.set_bad(color = 'k', alpha = 1.)

        # Loop over axes
        for i, ax in enumerate(axes_all):
            # Initial time step masked
            if i < 2:
                im = ax.imshow(torch.mean(self.losses_section[:,:,i+1], 1).unsqueeze(1), 
                    vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap)
            else:
                im = ax.imshow(torch.mean(self.losses_section[:,:,i+1], 1).unsqueeze(1), 
                    vmin=self.min_values[i+1], vmax=self.max_values[i+1], norm=self.define_log(log), cmap=cmap_e)

            # Create colorbars and orientate ticks
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="20%", pad=0.05)
            cbar1 = ax.figure.colorbar(im, ax=ax, cax=cax)
            plot_ticks(ax, ['Mean'], self.test_cases_reduced)

            # Ax format
            #ax.set_xlabel('Spatial Resolution')
            ax.set_title('{}'.format(self.losses[i]))

        # Save figure and 3 arrays
        fig.savefig(self.dir / 'MSE_mean.png', bbox_inches='tight')
        plt.close()