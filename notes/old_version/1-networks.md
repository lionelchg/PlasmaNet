# Directory: `outputs/1-networks/`

# Overview of the parameter study 
Influence of the number of layers: 3, 5, 7 (i.e. go up to 2^num_layers
divide of resolution) for 200 000 parameters
Influence of the number of parameters: 200 000, 400 000, 600 000 for 5 layers.
Investigate the link between the number of layers and the frequency content of the 
dataset. There seems to be a precision issue as $n = 10^{11}$ and not $n=10^{16}$ as
before. This impacts all the values as they are much more small in absolute value and the data processed by torch are float32. 

Add a scaling factor paramter (1e+6 for our cases).

We take a greedy approach for every parameter study. Meaning at every parameter space
exploration the best model is used to study further the network.

# Multi-scale

## Number of scales 
All the networks have parameters around 417 000 and it is in scales/ directory.
### MSNet3
```python
def __init__(self, data_channels):
    super(MSNet3, self).__init__()
    self.conv_4 = _ConvBlock1(data_channels, 32, 64, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 128, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### MSNet4
```python
def __init__(self, data_channels):
    super(MSNet5, self).__init__()
    self.conv_8 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 96, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### MSNet5
```python
def __init__(self, data_channels):
    super(MSNet5, self).__init__()
    self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
    self.conv_8 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 32, 48, 96, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### MSNet6
```python
def __init__(self, data_channels):
    super(MSNet7, self).__init__()
    self.conv_32 = _ConvBlock1(data_channels + 1, 16, 32, 1)
    self.conv_16 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_8 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 32, 64, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 32, 52, 96, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 112, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### Results on random_8 dataset
There is a huge gap between MSNet3 and the other three networks (MSNet[4-5-6]). MSNet5
seems to yield better results than MSNet4 and 6. Two questions arise:
- Why is the n / 8 scale (the added scale from MSNet3 to 4) so important?
- Why is MSNet5 an optimum in the three networks?

![alt text](../../NNet/pproc/figures/MSNet/scales/losses.png "Losses")
![alt text](../../NNet/pproc/figures/MSNet/scales/metrics.png "Residuals")


## Number of parameters
Among the previous networks MSNet5 seems to be the most promising. We start from these 5 scales and
change the number of parameters.

### MSNet5_small
```python
def __init__(self, data_channels):
    super(MSNet5_small, self).__init__()
    self.conv_16 = _ConvBlock1(data_channels, 20, 40, 1)
    self.conv_8 = _ConvBlock1(data_channels + 1, 20, 40, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 20, 40, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 20, 40, 80, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 20, 40, 80, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### MSNet5_big
```python
def __init__(self, data_channels):
    super(MSNet5_big, self).__init__()
    self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
    self.conv_8 = _ConvBlock1(data_channels + 1, 50, 100, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 54, 110, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 112, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### Results on random_8 dataset
As the number of parameters is increased the precision of the network is better.
The gap from MSNet5_small to MSNet5 is however bigger than from MSNet5 to MSNet5_big.
Let us now try to increase the number of parameters one step further and to add only
in the small scales / big scales. Both networks have around 900 000 parameters.


### MSNet5_big_1
```python
def __init__(self, data_channels):
    super(MSNet5_big, self).__init__()
    self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
    self.conv_8 = _ConvBlock1(data_channels + 1, 96, 144, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 96, 144, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 32, 64, 112, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 32, 64, 128, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### MSNet5_big_2
```python
def __init__(self, data_channels):
    super(MSNet5_big, self).__init__()
    self.conv_16 = _ConvBlock1(data_channels, 32, 64, 1)
    self.conv_8 = _ConvBlock1(data_channels + 1, 50, 100, 1)
    self.conv_4 = _ConvBlock1(data_channels + 1, 54, 110, 1)
    self.conv_2 = _ConvBlock2(data_channels + 1, 48, 96, 128, 1)
    self.conv_1 = _ConvBlock3(data_channels + 1, 48, 96, 144, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

### Results on random_8 dataset
MSNet_big_1 seems to yield better results than MSNet_big_2 so adding parameters to the upper scale
is more efficient than adding parameters to the down scales but the number of epochs seems to be
too low to get a real trend. Restart case maybe. For now on the best network is MSNet5_big

![alt text](../../NNet/pproc/figures/MSNet/nparams/losses.png "Losses")
![alt text](../../NNet/pproc/figures/MSNet/nparams/metrics.png "Residuals")



### Results on random_4, random_16, fourier_5, fourier_5_2, fourier_5_4 and target_case of MSNet5_big
|Dataset|Remarks|
|-------|-------|
|random_4| Results very close to random_8, slightly better on the losses (+overfitting with loss of valid that goes up)|
|random_16| The network struggles, the behavior is quite unexpected as there are less fine scales to understand|
|fourier_5| The network struggles, maybe the high value of the high frequencies is hard to capture|
|fourier_5_2| Better behavior compared to fourier_5 (expected)|
|fourier_5_4| Better behavior compared to fourier_5_2 (expected)|
|target_case| Very small loss the network learns very good the correlation (simple case as it is symmetric)|

![alt text](../../NNet/pproc/figures/MSNet/MSNet5_big/losses_fourier.png "Losses")
![alt text](../../NNet/pproc/figures/MSNet/MSNet5_big/metrics_fourier.png "Residuals")

![alt text](../../NNet/pproc/figures/MSNet/MSNet5_big/losses_random.png "Losses")
![alt text](../../NNet/pproc/figures/MSNet/MSNet5_big/metrics_random.png "Residuals")


## Weights of the different losses
Everything is run on random_8 and MSNet5_big for now:
| Loss weights (Inside, Dirichlet, Laplacian) | Remarks |
| ------------ | -------- |
| (1.0, 1.0, 0.2) | Set used since the beginning |
| (0.0, 0.0, 1.0) | Opposite case (full_lapl) |
| (0.5, 0.5, 1.0) | More laplacian (weight_1) |
| (1.0, 1.0, 1.0) | More laplacian (weight_2) |

# UNet

## Number of scales 
All the networks have parameters around 417 000 and it is in scales/ directory.
### UNet3
```python
def __init__(self, data_channels):
    super(UNet3, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 64, 64)
    self.convN_2 = _ConvBlockDown(64, 64, 64)
    self.convN_3 = _ConvBlockDown(64, 76, 64)
    self.convN_4 = _ConvBlockUp(128, 64, 64)
    self.convN_5 = _ConvBlockUp(128, 64, 64)
    self.final = _ConvBlockOut(64, 1)
```

### UNet4
```python
def __init__(self, data_channels):
    super(UNet4, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 48, 48)
    self.convN_2 = _ConvBlockDown(48, 48, 48)
    self.convN_3 = _ConvBlockDown(48, 48, 48)
    self.convN_4 = _ConvBlockDown(48, 64, 64)
    self.convN_5 = _ConvBlockUp(112, 64, 64)
    self.convN_6 = _ConvBlockUp(112, 60, 48)
    self.convN_7 = _ConvBlockUp(96, 48, 48)
    self.final = _ConvBlockOut(48, 1)
```

### UNet5
```python
def __init__(self, data_channels):
    super(UNet5, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 32, 32)
    self.convN_2 = _ConvBlockDown(32, 32, 32)
    self.convN_3 = _ConvBlockDown(32, 32, 32)
    self.convN_4 = _ConvBlockDown(32, 48, 62)
    self.convN_5 = _ConvBlockDown(62, 60, 62)
    self.convN_6 = _ConvBlockUp(124, 64, 64)
    self.convN_7 = _ConvBlockUp(96, 64, 64)
    self.convN_8 = _ConvBlockUp(96, 32, 32)
    self.convN_9 = _ConvBlockUp(64, 32, 32)
    self.final = _ConvBlockOut(32, 1)
```

### UNet6
```python
def __init__(self, data_channels):
    super(UNet6, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 32, 32)
    self.convN_2 = _ConvBlockDown(32, 32, 32)
    self.convN_3 = _ConvBlockDown(32, 32, 32)
    self.convN_4 = _ConvBlockDown(32, 48, 48)
    self.convN_5 = _ConvBlockDown(48, 48, 48)
    self.convN_6 = _ConvBlockDown(48, 48, 48)
    self.convN_7 = _ConvBlockUp(96, 60, 48)
    self.convN_8 = _ConvBlockUp(96, 48, 48)
    self.convN_9 = _ConvBlockUp(80, 48, 48)
    self.convN_10 = _ConvBlockUp(80, 32, 32)
    self.convN_11 = _ConvBlockUp(64, 32, 32)
    self.final = _ConvBlockOut(32, 1)
```

### Results on random_8 dataset
The UNets are far better than the MSNet for the equivalent scale. There is a similar trend, when we go deeper into the scales the results are better.
There is an uncertainty as to which one is better between 5 and 6 (even moreso than with MSNet). Why is UNet way better than MSNet is something
that could be investigated.

![alt text](../../NNet/pproc/figures/UNet/scales/losses.png "Losses")
![alt text](../../NNet/pproc/figures/UNet/scales/metrics.png "Residuals")

## Number of parameters

### UNet5_small_1
```python
def __init__(self, data_channels):
    super(UNet5_small_1, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 20, 20)
    self.convN_2 = _ConvBlockDown(20, 20, 20)
    self.convN_3 = _ConvBlockDown(20, 20, 20)
    self.convN_4 = _ConvBlockDown(20, 40, 40)
    self.convN_5 = _ConvBlockDown(40, 40, 40)
    self.convN_6 = _ConvBlockUp(80, 40, 40)
    self.convN_7 = _ConvBlockUp(60, 40, 40)
    self.convN_8 = _ConvBlockUp(60, 20, 20)
    self.convN_9 = _ConvBlockUp(40, 20, 20)
    self.final = _ConvBlockOut(20, 1)
```

### UNet5_small
```python
def __init__(self, data_channels):
    super(UNet5_small, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 20, 20)
    self.convN_2 = _ConvBlockDown(20, 20, 20)
    self.convN_3 = _ConvBlockDown(20, 20, 20)
    self.convN_4 = _ConvBlockDown(20, 48, 62)
    self.convN_5 = _ConvBlockDown(62, 60, 62)
    self.convN_6 = _ConvBlockUp(124, 64, 64)
    self.convN_7 = _ConvBlockUp(84, 64, 64)
    self.convN_8 = _ConvBlockUp(84, 20, 20)
    self.convN_9 = _ConvBlockUp(40, 20, 20)
    self.final = _ConvBlockOut(20, 1)
```

### UNet5_big
```python
def __init__(self, data_channels):
    super(UNet5_big, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 40, 40)
    self.convN_2 = _ConvBlockDown(40, 40, 40)
    self.convN_3 = _ConvBlockDown(40, 40, 40)
    self.convN_4 = _ConvBlockDown(40, 48, 78)
    self.convN_5 = _ConvBlockDown(78, 78, 78)
    self.convN_6 = _ConvBlockUp(156, 78, 78)
    self.convN_7 = _ConvBlockUp(118, 78, 78)
    self.convN_8 = _ConvBlockUp(118, 40, 40)
    self.convN_9 = _ConvBlockUp(80, 40, 40)
    self.final = _ConvBlockOut(40, 1)
```

### UNet6_big
```python
def __init__(self, data_channels):
    super(UNet6_big, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 40, 40)
    self.convN_2 = _ConvBlockDown(40, 40, 40)
    self.convN_3 = _ConvBlockDown(40, 40, 40)
    self.convN_4 = _ConvBlockDown(40, 60, 60)
    self.convN_5 = _ConvBlockDown(60, 60, 60)
    self.convN_6 = _ConvBlockDown(60, 60, 60)
    self.convN_7 = _ConvBlockUp(120, 60, 60)
    self.convN_8 = _ConvBlockUp(120, 60, 60)
    self.convN_9 = _ConvBlockUp(100, 60, 60)
    self.convN_10 = _ConvBlockUp(100, 40, 40)
    self.convN_11 = _ConvBlockUp(80, 40, 40)
    self.final = _ConvBlockOut(40, 1)
```

### Results on random_8 dataset

There seems to be no real gain to add parameters for UNet. 430 000 seems to be an
optimum as there is a slight gap between UNet5 and UNet5_small but not that obvious. 
Let us see what the network does with very few parameters. (UNet5_small_1)

![alt text](../../NNet/pproc/figures/UNet/nparams/losses.png "Losses")
![alt text](../../NNet/pproc/figures/UNet/nparams/metrics.png "Residuals")

## Loss
Try case only with Laplacian and Energy loss after Points Loss with UNet5:

| Loss weights (Points, Laplacian, Energy) | Remarks |
| ------------ | --------|
| (1.0, 0.2, 0.0) | Classical case |
| (1.0, 0.0, 0.2) | Energy case (random_8_energy), the results seem good however the absolute value of the loss seem too low for float32 (1e-7 maybe it is too close to machine precision) |
| (1.0, 0.0, 1e8) | Energy case (random_8_energy_1) with higher weight for energy, the results don't change that much compared to random_8_energy. The results are good except for the electric field at the boundaries which seem to be bad |
| (0.0, 0.0, 1e8) | Only Energy case (random_8_energy_2). Bad results on eletric field |
| (0.0, 0.1, 1e8) | Laplacian and energy case (random_8_energy_3). Bad results on electric field |
| (0.0, 1.0, 0.2, 0.0) | Laplacian with Dirichlet only -> close to the system (random_8_energy_4). The results seem promising, the electric field seems to be globally better and focus on smoothness of solution may happen here. This is the way that the actual problem is solved physically, we impose potential on boundaries and impose the laplacian inside so the loss reflects that *Try a target_case on this*.|

So there is a need to have a good reference for the potential. Only having a DirichletLoss works in this case. Need to introduce metrics for the electric field because that is what we care about in the end for the simulation (even the divergence strictly speaking).

## Plasma oscillation

With a UNet5 (common loss weights at 1.0 / 1.0 / 0.2 for Inside/Dirichlet/Laplacian), the results are way better than the MSNet5_big.
