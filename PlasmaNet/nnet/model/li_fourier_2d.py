########################################################################################################################
#                                                                                                                      #
#                                               2D Fourier network from                                                #
#                       "Fourier Neural Operator for Parametric Partial Differential Equations"                        #
#                                by Zongyi Li et. al., https://arxiv.org/abs/2010.08895                                #
#                    https://github.com/zongyi-li/fourier_neural_operator/blob/master/fourier_2d.py                    #
#                                                                                                                      #
#                                Ekhi Ajuria, Guillaume Bogopolsky, CERFACS, 19.01.2021                                #
#                                                                                                                      #
########################################################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.nn.parameter import Parameter

import operator
from functools import partial, reduce


# Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

# Ensurer periodicity
def mirror(x):
    double_size = []
    # The input is supposed of dim 4 (bsz, channels, H, W)
    for i, size in enumerate(list(x.size())):
        if i < 2:
            double_size.append(size)
        # Only double up H and W
        else:
            double_size.append(int(2 * size -1))

    # Tensor and mesh_size declaration
    mirror = torch.zeros(double_size)
    mesh_size = x.size(3)

    # convert to numpy
    mirror_np = mirror.cpu().numpy()    
    x_np = x.cpu().numpy()

    # Mirroring
    mirror_np[:, :, mesh_size - 1 :, mesh_size - 1 :] = x_np[:,:]
    mirror_np[:, :, : mesh_size - 1, : mesh_size - 1] = x_np[:, :, -1:0:-1, -1:0:-1]
    mirror_np[:, :, mesh_size - 1 :, : mesh_size - 1] = - x_np[:, :, :, -1:0:-1]
    mirror_np[:, :, : mesh_size - 1, mesh_size - 1 :] = - x_np[:, :, -1:0:-1, :]

    # Reconvert to torch
    mirror = torch.from_numpy(mirror_np).cuda()
    x = torch.from_numpy(x_np).cuda()

    return mirror

# Not that useful, but just in case, takes simple and doubled up domains.
def unmirror(x_small, x_big):
    # Declaration os tensor and mesh size
    unmirrored = torch.zeros_like(x_small)
    mesh_size = x_small.size(3)

    # Take correct cuadrant!    
    unmirrored[:,:] = x_big[:,:, mesh_size - 1 :, mesh_size - 1 :]
    return unmirrored

################################################################
# Fourier layer
################################################################

class _SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(_SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.fft.rfft(x, 2, norm="forward")
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        # x = torch.fft.irfft(out_ft, 2, norm="forward", signal_sizes=( x.size(-2), x.size(-1)))
        return x


class _SimpleBlock2d(nn.Module):
    def __init__(self, data_channels, modes1, modes2,  width):
        super(_SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.data_channels = data_channels
        self.fc0 = nn.Linear(self.data_channels, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = _SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = _SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = _SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = _SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        # Double up the domain to ensure periodicity
        # save small domain just in case (not that necessary, but just in case)
        x_small = x
        x = mirror(x)

        batchsize = x.shape[0]
        size_x, size_y = x.shape[2], x.shape[3]

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)

        # Take the correct cuadrant!
        x = unmirror(x_small, x)

        return x


class FourierNet2D(nn.Module):
    """
    A wrapper class
    """
    def __init__(self, data_channels, modes, width):
        super(FourierNet2D, self).__init__()
        self.conv1 = _SimpleBlock2d(data_channels, modes, modes, width)

    def forward(self, x):
        x = self.conv1(x)
        # return x.squeeze()
        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c
