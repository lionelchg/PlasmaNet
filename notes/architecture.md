# Notes on architecture of scales networks

In the following, general definitions applying on scales network (encompassing both UNet and MSNet) are presented. The network is described in terms of depth and width. The global receptive field of the network is also defined.

## Depth of the network

The local depth $d$ is defined as the power of 2 by which the initial resolution is divided by in a specific scale of the network. The depth $D$ of the network is defined as the maximum power of the local depths. 

## Width of the network

The width $W$ of the network is defined as the total number of convolutional layers of the network. The higher the width of the network, the more likely vanishing or exploding gradients can appear.

$$
W = \sum_{d=0}^D W_d
$$

## Receptive field of the network

The receptive field of the scales network is defined as:

$$
\mrm{RF} = \sum_{d=0}^D \mrm{RF}_d
$$
with
$$
\mrm{RF}_d = W_d (k_s^d - 1) 2^d
$$
where $k_s^d$ is the kernel size of depth $d$.