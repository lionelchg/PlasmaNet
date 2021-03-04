import re
import numpy as np
import scipy.constants as co

import torch
from PlasmaNet.cfdsolver.euler.plasma import PlasmaEuler
from PlasmaNet.cfdsolver.scalar.streamer import StreamerMorrow


class StreamerMorrowDL(StreamerMorrow):
    """ Solve poisson with PlasmaNet. """
    def __init__(self, config):
        super().__init__(config)
        self.alpha = 0.1
        self.ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / self.Lx**2 + 1 / self.Ly**2)

    def solve_poisson_dl(self, model):
        """ Solve poisson equation with PlasmaNet. """
        self.physical_rhs = (self.nd[1] - self.nd[0] - self.nd[2]) * co.e / co.epsilon_0
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                                * self.ratio).float().cuda()
        potential_torch = model(physical_rhs_torch)
        potential_rhs = potential_torch.detach().cpu().numpy()[0, 0]
        self.potential = potential_rhs - self.backE * self.X


class PlasmaEulerDL(PlasmaEuler):
    """ Solve poisson with PlasmaNet. """
    def __init__(self, config, config_dl):
        super().__init__(config)
        self.alpha = 0.1
        self.ratio = self.alpha / (np.pi**2 / 4)**2 / (1 / self.Lx**2 + 1 / self.Ly**2)
        self.scaling_factor = 1.0e+6
        self.res_sim = config['mesh']['nnx']
        self.res_train = config_dl['globals']['nnx']

        if hasattr(self, 'globals'):
            self.globals['nnx_nn'] = self.res_train
            self.globals['Lx_nn'] = config_dl['globals']['lx']
            self.globals['arch'] = config_dl['arch']['type']
            
            re_casename = re.compile(r'.*/(\w*)/(\w*)/(\w*)')
            if re_casename.search(config_dl['resume']):
                self.globals['train_dataset'] = re_casename.search(config_dl['resume']).group(1)
            

    def solve_poisson_dl(self, model):
        """ Solve poisson equation with neural network model (pytorch object) """
        self.physical_rhs = - (self.U[0] / self.m_e - self.n_back) * co.e / co.epsilon_0
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                                * self.ratio * self.scaling_factor).float().cuda()
        potential_torch = model(physical_rhs_torch)
        potential_rhs = (self.res_train**2 / self.res_sim**2 * potential_torch.detach().cpu().numpy()[0, 0] 
                                                / self.scaling_factor)
        
        self.poisson.potential = potential_rhs
        self.E_field = self.poisson.E_field
    
        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1: self.E_max = np.max(self.E_norm)

class PlasmaEulerDLMSE(PlasmaEulerDL):
    """ Solve poisson with PlasmaNet. """
    def __init__(self, config, config_dl):
        super().__init__(config, config_dl)
        # Declare variable for postproc
        self.physical_MSE_rhs_list = np.zeros((self.nit, self.nny, self.nnx))

    def solve_poisson_dl(self, model):
        """ Solve poisson equation with PlasmaNet. """
        self.physical_rhs = - (self.U[0] / self.m_e - self.n_back) * co.e / co.epsilon_0
        self.physical_MSE_rhs_list[self.it -1] = self.physical_rhs 
        # Convert to torch.Tensor of shape (batch_size, 1, H, W) with normalization
        physical_rhs_torch = torch.from_numpy(self.physical_rhs[np.newaxis, np.newaxis, :, :] 
                                                * self.ratio * self.scaling_factor).float().cuda()
        potential_torch = model(physical_rhs_torch)
        potential_rhs = (self.res_train**2 / self.res_sim**2 * potential_torch.detach().cpu().numpy()[0, 0] 
                                                / self.scaling_factor)
        
        self.poisson.potential = potential_rhs
        self.E_field = self.poisson.E_field
    
        self.E_norm = np.sqrt(self.E_field[0]**2 + self.E_field[1]**2)
        if self.it == 1: self.E_max = np.max(self.E_norm)

    def plot_MSE(self, dataset_potential, dataset_rhs, perc):
        """ Plot 3 subplots:
        * MSE evolution 
        * Temporal deviation from a cosinus evolution of the (100-perc)% internals 
        * MSE of the perc% external values
        Everything normalized to plasma period units.
        Apply the normalization on temporals as well 
        Note: Use small perc values (~5) """
        # Normalize the rhs fields by the initial dataset max value!
        physical_rhs_list = - self.physical_MSE_rhs_list * co.epsilon_0 / co.e 
        physical_rhs_list /= np.max(dataset_rhs[0])
        dataset_rhs /= np.max(dataset_rhs[0]) 

        # For plotting
        time_plot = self.time / self.T_p 
        omega_p_plot = self.omega_p * self.T_p

        # Calculate the MSE evolution of the entire simulation and the temporal evolution
        # of the entire field
        global_MSE = np.sum((physical_rhs_list-dataset_rhs)**2, axis=(1,2))/(self.nnx **2)
        temp_global = np.sum(physical_rhs_list, axis=(1,2))/(self.nnx **2)
        temp_global /= temp_global[0] 
        diff_temp_global = np.abs(temp_global - np.cos(omega_p_plot * time_plot))

        # Use a mask to differentiate perc% in and (100-perc)% out when t = 0
        # Calculate limit and the number of smaller and bigeer points 
        limit_value = np.percentile(dataset_rhs[0], perc)
        mask_max = np.where(dataset_rhs[0] > limit_value, 1.0, 0.0)
        mask_min = np.where(dataset_rhs[0] <= limit_value, 1.0, 0.0)
        mask_plot_all = mask_max > -0.5  
        mask_plot_max = mask_max > 0.5 
        mask_plot_min = mask_max < 0.5
        max_pointn = np.count_nonzero(mask_max)
        min_pointn = np.count_nonzero(mask_min) 

        # Mask the respective arrays
        dataset_max_values = dataset_rhs * mask_max
        predicted_max_values = physical_rhs_list * mask_max 
        dataset_small_values = dataset_rhs * mask_min 
        predicted_small_values = physical_rhs_list * mask_min

        # Temporal evolution and MSE of the Mean of max and min predicted values
        # Normalized by the initial value of the dataset
        # Substract baseline cosinus
        max_MSE = np.sum((predicted_max_values-dataset_max_values)**2, axis=(1,2))/max_pointn
        temp_max = np.sum(predicted_max_values, axis=(1,2))/max_pointn
        temp_max /= (np.sum(dataset_max_values, axis=(1,2))/max_pointn)[0]
        diff_temp_max = np.abs(temp_max - np.cos(omega_p_plot * time_plot))

        min_MSE = np.sum((predicted_small_values-dataset_small_values)**2, axis=(1,2))/min_pointn
        temp_min_mean = np.sum(predicted_small_values, axis=(1,2))/min_pointn
        temp_min_mean/= (np.sum(dataset_max_values, axis=(1,2))/max_pointn)[0] 
        
        # gridspec inside gridspec
        fig = plt.figure(constrained_layout=True, figsize=(10, 8))

        gs = GridSpec(3, 3, figure=fig)

        ax0a = fig.add_subplot(gs[0, 0])
        ax0b = fig.add_subplot(gs[1, 0])
        ax0c = fig.add_subplot(gs[2, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])
        ax4 = fig.add_subplot(gs[0, 2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[2, 2])

        # Initial time step masked
        ax0a.imshow(~mask_plot_all, alpha=0.2, cmap='gray') 
        ax0b.imshow(~mask_plot_max, alpha=0.2, cmap='gray')
        ax0c.imshow(~mask_plot_min, alpha=0.2, cmap='gray') 

        # Plots for Global Field
        ax1.plot(time_plot, global_MSE, 'k', label='MSE')
        ax4.plot(time_plot, temp_global, 'k', label='Temporal Evolution')
        ax4.plot(time_plot, np.cos(omega_p_plot * time_plot), 'k', linestyle='dashed', label='Baseline') 
        ax4bis = ax4.twinx()
        ax4bis.plot(time_plot, diff_temp_global, 'k', linestyle='dotted', label='Temporal error')

        # Plots for Max Field
        ax2.plot(time_plot, max_MSE, 'k', label='MSE')
        ax5.plot(time_plot, temp_max, 'k', label='Temporal Evolution')
        ax5.plot(time_plot, np.cos(omega_p_plot * time_plot), 'k', linestyle='dashed', label='Baseline') 
        ax5bis = ax5.twinx()
        ax5bis.plot(time_plot, diff_temp_max, 'k', linestyle='dotted', label='Temporal error')

        # Plots for Min Field
        ax3.plot(time_plot, min_MSE, 'k', label='MSE')
        ax6.plot(time_plot, np.abs(temp_min_mean), 'k', label='Temporal Evolution')

        fig.suptitle("Error Evolution")

        # Ax format
        # Global
        self.ax_prop(ax1, '$t / T_p$', r'Global MSE', None, legend=False)
        ax1.set_ylim(bottom=0.000001)
        ax1.set_yscale('log')
        self.ax_prop(ax4, '$t / T_p$', r'Global mean evolution', None, legend=False)
        ax4bis.set_ylim(bottom=0.000001)
        ax4bis.set_yscale('log')
        ax4bis.set_ylabel('Temporal Evolution Error')          
        # Max
        self.ax_prop(ax2, '$t / T_p$', r'Inside value MSE', None, legend=False)
        ax2.set_ylim(bottom=0.000001)
        ax2.set_yscale('log')
        self.ax_prop(ax5, '$t / T_p$', r'Inside value mean evolution', None, legend=False)
        ax5bis.set_ylim(bottom=0.000001)
        ax5bis.set_yscale('log')
        ax5bis.set_ylabel('Temporal Evolution Error')
        # Min
        self.ax_prop(ax3, '$t / T_p$', r'Outside value MSE', None, legend=False)
        ax3.set_ylim(bottom=0.000001)
        ax3.set_yscale('log')
        self.ax_prop(ax6, '$t / T_p$', r'Outside value mean evolution', None, legend=False)
        # Masks
        ax0a.set_axis_off()
        ax0a.set_title('Entire Domain')
        ax0b.set_axis_off()
        ax0b.set_title('Inside {}% values'.format(100-self.perc)) 
        ax0c.set_axis_off()
        ax0c.set_title('Outside {}% values'.format(self.perc)) 

        # Save figure and 3 arrays
        fig.savefig(self.fig_dir + 'MSE', bbox_inches='tight')
        np.save(self.case_dir + 'global_MSE.npy', global_MSE)
        np.save(self.case_dir + 'temp_max.npy', temp_max)
        np.save(self.case_dir + 'min_MSE.npy', min_MSE)


    def post_MSE(self, data_dir_base):
        """ Postproc to plot and save MSE. """ 
        # For now let's set it manually
        self.perc = 30

        # Get corresponding directory
        parent = os.path.dirname(os.path.dirname(os.path.dirname(data_dir_base)))
        data_dir_case = os.path.join(parent, '{}.{}'.format(self.nnx, self.nnx), 
            '{}_periods'.format(np.int(self.n_periods)), 'dl_data') 

        # Load Potential 
        dataset_potential = np.load(os.path.join(data_dir_case, "potential.npy"))
        dataset_rhs = - np.load(os.path.join(data_dir_case, "physical_rhs.npy")) * co.epsilon_0 / co.e 

        # Plot and save the results
        self.plot_MSE(dataset_potential, dataset_rhs, self.perc) 