########################################################################################################################
#                                                                                                                      #
#                                                    Loss classes                                                      #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 03.03.2020                                #
#                                                                                                                      #
########################################################################################################################

import sys
import os

import scipy.constants as co
import torch
import torch.nn.functional as F
import copy

import numpy as np
import matplotlib.pyplot as plt

from ..base.base_loss import BaseLoss
from ..operators.gradient import gradient_scalar
from ..operators.laplacian import laplacian as lapl
from ..utils.util import initialize_figures, plot_kde, save_gradient_plots
from ...poissonscreensolver.photo import A_j_two, lambda_j_two, pO2

class InsideLoss(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). """
    def __init__(self, config, inside_weight, **_):
        super().__init__()
        self.weight = inside_weight
        self.base_weight = self.weight
        self.cyl = config.coord == 'cyl'
        self.label = 'Inside'

    def _forward(self, output, target, **kwargs):
        if self.cyl:
            return F.mse_loss(output[:, 0, 0:-1, 1:-1], target[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return F.mse_loss(output[:, 0, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1]) * self.weight

class AxialDirichletLoss(BaseLoss):
    """ Computes the weighted MSELoss of the BC in the axis (for the cylindrical case). """
    def __init__(self, config, axial_dir_weight, **_):
        super().__init__()
        self.weight = axial_dir_weight
        self.base_weight = self.weight
        self.label = 'Axial Dirichlet'

    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 0, 0, 1:-1], target[:, 0, 0, 1:-1]) * self.weight

class AxialNeumannLoss(BaseLoss):
    """ Loss for Neumann boundary at the axis enforcing \nabla \phi \cdot \vb{n} = 0. """
    def __init__(self, config, axial_neu_weight, **_):
        super().__init__()
        self.weight = axial_neu_weight
        self.base_weight = self.weight
        self.dx = config.dx
        self.dy = config.dy
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.label = 'Axial Neumann'

    def _forward(self, output, target, **_):
        # Compute normal electric field at the axis
        grad_output = (4 * output[:, 0, 1, 1:-1] - 3 * output[:, 0, 0, 1:-1] - output[:, 0, 2, 1:-1]) / (2 * self.dy)

        # Loss on that boundary
        bnd_loss = F.mse_loss(grad_output, torch.zeros_like(grad_output))

        return self.Lx * self.Ly * bnd_loss * self.weight

class MSInsideLoss_n(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). For the n scale"""
    def __init__(self, config, inside_weight_n, **_):
        super().__init__()
        self.weight_n = inside_weight_n
        self.base_weight = self.weight_n
        self.label = 'Inside n'
    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 1, 1:-1, 1:-1], target[:, 0, 1:-1, 1:-1]) * self.weight_n

class MSInsideLoss_n2(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). For the n2 scale"""
    def __init__(self, config, inside_weight_n2, **_):
        super().__init__()
        self.weight_n2 = inside_weight_n2
        self.label = 'Inside n2'
    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 2, 1:-1, 1:-1], target[:, 1, 1:-1, 1:-1]-target[:, 2, 1:-1, 1:-1]) * self.weight_n2

class MSInsideLoss_n4(BaseLoss):
    """ Computes the weighted MSELoss of the interior of the domain (excluding boundaries). For the n4 scale"""
    def __init__(self, config, inside_weight_n4, **_):
        super().__init__()
        self.weight_n4 = inside_weight_n4
        self.label = 'Inside n4'
    def _forward(self, output, target, **kwargs):
        return F.mse_loss(output[:, 3, 1:-1, 1:-1], target[:, 2, 1:-1, 1:-1]) * self.weight_n4


class LaplacianLoss(BaseLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, config, lapl_weight, **_):
        super().__init__()
        self.weight = lapl_weight
        self.base_weight = self.weight
        self.dx = config.dx
        self.dy = config.dy
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.r_nodes = config.r_nodes
        if self.r_nodes is not None:  # in this case, a NumPy array
            self.r_nodes = torch.from_numpy(self.r_nodes).cuda()
        self._require_input_data = True  # Need rhs for computation
        self.cyl = config.coord == 'cyl'
        self.label = 'Laplacian'

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        laplacian = lapl(output * target_norm / data_norm, self.dx, self.dy, r=self.r_nodes)
        if self.cyl:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 0:-1, 1:-1], - data[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 0, 1:-1, 1:-1]) * self.weight

class PhotoLoss_j1(LaplacianLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, config, photo_weight_j1, **_):
        super().__init__(config, photo_weight_j1)

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        # Renormalize to make it coherent with RHS
        out_norm = output / data_norm
        laplacian = lapl(out_norm, self.dx, self.dy, r=self.r_nodes)
        if self.cyl:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 0:-1, 1:-1]
            - (lambda_j_two[0] * pO2)**2 * out_norm[:, 0, 0:-1, 1:-1],
            - A_j_two[0] * pO2**2 * data[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1],
            - (lambda_j_two[0] * pO2)**2 * out_norm[:, 0, 1:-1, 1:-1],
            - A_j_two[0] * pO2**2 * data[:, 0, 1:-1, 1:-1]) * self.weight

class PhotoLoss_j2(LaplacianLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, config, photo_weight_j2, **_):
        super().__init__(config, photo_weight_j2)

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        # Renormalize to make it coherent with RHS
        out_norm = output / data_norm
        laplacian = lapl(out_norm, self.dx, self.dy, r=self.r_nodes)
        if self.cyl:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 0:-1, 1:-1]
            - (lambda_j_two[1] * pO2)**2 * out_norm[:, 0, 0:-1, 1:-1],
            - A_j_two[1] * pO2**2 * data[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1],
            - (lambda_j_two[1] * pO2)**2 * out_norm[:, 0, 1:-1, 1:-1],
            - A_j_two[1] * pO2**2 * data[:, 0, 1:-1, 1:-1]) * self.weight

class PhotoLoss(LaplacianLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries). """
    def __init__(self, config, photo_weight, **_):
        super().__init__(config, photo_weight)
        self.photo_scale = config.config['data_loader']['args']['lambda_scale']

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        # Renormalize to make it coherent with RHS
        out_norm =  output / data_norm
        #data *= self.photo_scale
        #data[:,0] *= data_norm[:,0]
        data[:,1] *= self.photo_scale
        #laplacian_norm = lamb**2 + 1/(self.dx*self.dy)
        #out_norm =  output * data_norm/ laplacian_norm
        laplacian = lapl(out_norm, self.dx, self.dy, r=self.r_nodes)

        lamb = torch.mean(data[:, 1])
        rhs_photo_scale = 1.0 * (lamb**2 + 1/(self.dx*self.dy))/ data_norm.mean()

        #rhs_photo_scale = 1.0 #0.572*(lamb**2) - 512.32*lamb + 1
        if self.cyl:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 0:-1, 1:-1]
            - data[:, 1, 0:-1, 1:-1]**2 * out_norm[:, 0, 0:-1, 1:-1],
            - rhs_photo_scale * data[:, 0, 0:-1, 1:-1]) * self.weight
        else:
            return self.Lx**2 * self.Ly**2 * F.mse_loss(laplacian[:, 0, 1:-1, 1:-1],
            - data[:, 1, 1:-1, 1:-1]**2 * out_norm[:, 0, 1:-1, 1:-1],
            - rhs_photo_scale * data[:, 0, 1:-1, 1:-1]) * self.weight

class EnergyLoss(BaseLoss):
    """ An Energy loss that is minimum for Poisson's equation (Variational approach). """
    def __init__(self, config, energy_weight, **_):
        super().__init__()
        self.weight = energy_weight
        self.base_weight = self.weight
        self.dx = config.dx
        self.dy = config.dy
        self.n_inputs = config.nnx * config.nny * config.batch_size
        self._require_input_data = True  # Need rhs for computation
        self.label = 'Energy'

    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        elec_output = gradient_scalar(output, self.dx, self.dy)
        # Need to use data_norm because the rhs has been changed due to scaling
        norm_elec_output = (elec_output[:, 0, :, :] ** 2 + elec_output[:, 1, :, :] ** 2).unsqueeze(1) / data_norm
        energy_output = (0.5 * norm_elec_output - output * data)
        return co.epsilon_0 * torch.sum(energy_output) / self.n_inputs * self.weight


class ElectricLoss(BaseLoss):
    """ Loss function on the electric field. """
    def __init__(self, config, elec_weight, **_):
        super().__init__()
        self.weight = elec_weight
        self.base_weight = self.weight
        self.dx = config.dx
        self.dy = config.dy
        self._require_input_data = False
        self.label = 'Electric'

    def _forward(self, output, target, target_norm=1., data_norm=1., **_):
        elec_output = gradient_scalar(output * target_norm / data_norm, self.dx, self.dy)
        elec_target = gradient_scalar(target * target_norm / data_norm, self.dx, self.dy)
        return F.mse_loss(elec_output, elec_target) * self.weight


class DirichletBoundaryLoss(BaseLoss):
    """ Loss for Dirichlet boundaries. self.cyl boolean to determine if the y = 0 boundary is dirichlet or not """
    def __init__(self, config, bound_weight, **_):
        super().__init__()
        self.weight = bound_weight
        self.base_weight = self.weight
        self.cyl = config.coord == 'cyl'
        self.label = 'Dirichlet BC'

    def _forward(self, output, target, **_):
        bnd_loss = F.mse_loss(output[:, 0, -1, :], torch.zeros_like(output[:, 0, -1, :]))
        bnd_loss += F.mse_loss(output[:, 0, :, 0], torch.zeros_like(output[:, 0, :, 0]))
        bnd_loss += F.mse_loss(output[:, 0, :, -1], torch.zeros_like(output[:, 0, :, -1]))
        # Axis points to add only if we're in cartesian coordinates
        if not self.cyl: bnd_loss += F.mse_loss(output[:, 0, 0, :], torch.zeros_like(output[:, 0, 0, :]))
        return bnd_loss * self.weight


class NeumannBoundaryLoss(BaseLoss):
    """ Loss for Neumann boundaries. """
    def __init__(self, config, bound_weight, **_):
        super().__init__()
        self.weight = bound_weight
        self.base_weight = self.weight
        self.dx = config.dx
        self.dy = config.dy
        self.label = 'Neumann BC'

    def _forward(self, output, target, **_):
        # Compute electric field
        grad_output = gradient_scalar(output, self.dx, self.dy)
        grad_target = gradient_scalar(target, self.dx, self.dy)
        # Loss on the boundaries
        bnd_loss = F.mse_loss(grad_output[:, 0, 1:-1, 0], grad_target[:, 0, 1:-1, 0])  # line for x = 0
        bnd_loss += F.mse_loss(grad_output[:, 0, 1:-1, -1], grad_target[:, 0, 1:-1, -1])
        bnd_loss += F.mse_loss(grad_output[:, 1, 0, 1:-1], grad_target[:, 1, 0, 1:-1])
        bnd_loss += F.mse_loss(grad_output[:, 1, -1, 1:-1], grad_target[:, 1, -1, 1:-1])
        return bnd_loss * self.weight


class LongTermLaplacianLoss(LaplacianLoss):
    """ A Laplacian loss function on the inside of the domain (excluding boundaries),
    for a long term loss. """
    def __init__(self, config, lt_weight, **_):
        super().__init__(config, lt_weight)
    def _forward(self, output, target, data=None, target_norm=1., data_norm=1., **_):
        laplacian = lapl(output[:,1].unsqueeze(1) * target_norm / data_norm, self.dx, self.dy, r=self.r_nodes)
        return F.mse_loss(laplacian[:, 0, 1:-1, 1:-1], - data[:, 1, 1:-1, 1:-1]) * self.weight


class ComposedLoss(BaseLoss):
    """
    Class for loss composition, with list of losses to compose as argument.
    """
    def __init__(self, config, loss_list, **kwargs):
        """
        Initializes the ComposedLoss with the loss specified in loss_list, and pass the arguments by dictionary
        from the input file.
        Therefore, the loss must have different argument name for the weight for example.
        """
        super().__init__()
        # Initializes all losses in a list
        self.loss_list = loss_list
        self.losses = list()
        for loss in self.loss_list:
            try:
                self.losses.append(getattr(sys.modules[__name__], loss)(config, **kwargs))  # Look for classes in this module
            except TypeError as err:
                print('while initializing {} class:'.format(loss))
                raise type(err)('{} while initializing {} class.'.format(err, loss))
        # Check if any loss requires data input
        self._require_input_data = any([loss.require_input_data() for loss in self.losses])
        # Store results for logger access
        self.results = torch.zeros(len(self.loss_list))  # Use numpy to ensure floating-point sum
        # Save folder for gradient plotting
        self.fig_dir = config._fig_dir
        # If gradient plotting, plot every n epochs, else plot every 0!
        if config.gradients:
            self.gradients_every = config.gradients_every
        else:
            self.gradients_every = 0

    def _forward(self, output, target, **kwargs):
        out = self.losses[0](output, target, **kwargs)
        self.results[0].copy_(out)
        for i, loss in enumerate(self.losses[1:]):
            tmp = loss(output, target, **kwargs)
            out += tmp
            self.results[i + 1].copy_(tmp)
        return out

    def intermediate(self, model, optimizer, output, target, epoch, ibatch, **kwargs):
        """
        Calculates the mean and max gradients of all the forwarded losses.
        """
        # Initialize lists that will contain the max and mean grads
        max_grads = []
        mean_grads = []

        # Only perform plots if plot gradients is not 0
        plot_gradients = (self.gradients_every > 0) and (epoch % self.gradients_every == 0) and (ibatch == 0)

        # TODO generalizing for specific gradients
        # Initialize lists to save the gradients of each loss individually
        #gradients_lapl = []
        #gradients_rest = []
        #gradients_bc_ax = []

        # Initialize 4 plots corresponding to last, -1, +1, 0
        if plot_gradients:
            fig_list, ax_list = initialize_figures(4)
            fig_list_norm, ax_list_norm = initialize_figures(4)

        for loss in self.losses:
            # Compute each individual loss and retain graph as multiple backwards will be performed.
            loss_indv = loss(output, target, **kwargs)
            loss_indv.backward(retain_graph=True)

            # Initialize list containing the gradients of each weight and the max/mean/n_el values
            # As the list contains elements with variable sizes, the number of weights of each layer
            # is stored in the n_el variable to correctly compute the mean
            gradients = []
            max_grad = 0
            mean_grad = 0
            n_el = 0

            # Loop over the network variables, storing the max grad
            # and adding the sum of each layer, as well as the number of elements
            for p in model.parameters():
                gradients.append(p.grad)

                # TODO generalizing for specific gradients
                #if isinstance(loss, LaplacianLoss):
                #    gradients_lapl.append(p.grad.clone().detach())
                #elif isinstance(loss, DirichletBoundaryLoss):
                #    gradients_bc_dr.append(p.grad.clone().detach())
                #elif isinstance(loss, InsideAxialLoss):
                #    gradients_bc_ax.append(p.grad.clone().detach())

                if torch.max(torch.abs(p.grad)) > max_grad:
                    max_grad = torch.max(torch.abs(p.grad))
                mean_grad +=torch.sum(torch.abs(p.grad))
                n_el += len(p.grad.view(-1))

            # Compute overall mean of the gradients
            mean_grad /= n_el

            # Store the computed values
            max_grads.append(max_grad)
            mean_grads.append(mean_grad)

            # Plot
            if plot_gradients:

                # plot k_de for not normalized gradients for the:
                # last layer (output), m1 layer (minus 1), p1 layer (plus 1), zero (input)
                x = np.linspace(-2*max_grad.detach().cpu().numpy(), 2*max_grad.detach().cpu().numpy(), 200)
                plot_kde(x, gradients[-2], ax_list[0], loss.label)
                plot_kde(x, gradients[-4], ax_list[1], loss.label)
                plot_kde(x, gradients[0], ax_list[2], loss.label)
                plot_kde(x, gradients[2], ax_list[3], loss.label)

                # plot k_de for normalized gradients for the:
                # last layer (output), m1 layer (minus 1), p1 layer (plus 1), zero (input)
                xn = np.linspace(-1, 1, 200)
                plot_kde(xn, gradients[-2]/max_grad, ax_list_norm[0], loss.label)
                plot_kde(xn, gradients[-4]/max_grad, ax_list_norm[1], loss.label)
                plot_kde(xn, gradients[0]/max_grad, ax_list_norm[2], loss.label)
                plot_kde(xn, gradients[2]/max_grad, ax_list_norm[3], loss.label)

            # Clean the gradients to avoid overlap!
            optimizer.zero_grad()


        if plot_gradients:
            # TODO Generalize to count number of greater gradients ...
            # Check which gradients are smaller ! Start initializing
            #values = 0
            # Loop over network layers
            #for i in range(len(gradients_lapl)):
            #    # Check maximum gradient value between axial and dirichlet losses
            #    max_bc = torch.where(torch.abs(gradients_bc_dr[i]) > torch.abs(gradients_bc_ax[i]),
            #                        gradients_bc_dr[i], gradients_bc_ax[i])
            #    # Check where the laplacian gradients are higher
            #    lapl_big = torch.where(torch.abs(gradients_lapl[i]) > torch.abs(max_bc),
            #                        torch.ones_like(gradients_lapl[i]), torch.zeros_like(gradients_lapl[i]))
            #    val = torch.sum(lapl_big).detach().cpu()
            #    #if val >0:
            #    #    print('There were {} learning weights found in layer: {} with a shape of: {}'.format(int(val), i, gradients_lapl[i].shape))
            #    values += val
            #print("Number of laplacian values learning something: ", int(values))

            # Add legend and save plots with raw and normalized gradients, and close images to avoid memory leak
            name_list = ['_last.png', '_m_1.png', '_zero.png', '_p_1.png']
            save_gradient_plots(fig_list, ax_list, name_list, self.fig_dir, epoch)
            name_list = ['_last_normalized.png', '_m_1_normalized.png', '_zero_normalized.png', '_p_1_normalized.png']
            save_gradient_plots(fig_list_norm, ax_list_norm, name_list, self.fig_dir, epoch)
            plt.close('all')

            # # Useful print?
            # print('Max grads: ', max_grads)
            # print('Mean grads: ', mean_grads)

        return max_grads, mean_grads

    def log(self):
        """ Returns log of each loss independently as a dict() with loss names keys and the associated torch values. """
        out = {loss_name: result for loss_name, result in zip(self.loss_list, self.results)}
        out.update({'loss': self.result})
        return out
