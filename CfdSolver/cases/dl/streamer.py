import numpy as np
import scipy.constants as co
import torch
from cfdsolver import StreamerMorrow


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
