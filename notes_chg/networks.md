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

![alt text](figures/network_training/scales/losses.png "Losses")
![alt text](figures/network_training/scales/metrics.png "Residuals")


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

![alt text](figures/network_training/nparams/losses.png "Losses")
![alt text](figures/network_training/nparams/metrics.png "Residuals")



### Results on random_4, random_16, fourier_5, fourier_5_2, fourier_5_4 and target_case of MSNet5_big
|Dataset|Remarks|
|-------|-------|
|random_4| Results very close to random_8, slightly better on the losses (+overfitting with loss of valid that goes up)|
|random_16| The network struggles, the behavior is quite unexpected as there are less fine scales to understand|
|fourier_5| The network struggles, maybe the high value of the high frequencies is hard to capture|
|fourier_5_2| Better behavior compared to fourier_5 (expected)|
|fourier_5_4| Better behavior compared to fourier_5_2 (expected)|
|target_case| Very small loss the network learns very good the correlation (simple case as it is symmetric)|

![alt text](figures/network_training/MSNet5_big/losses_fourier.png "Losses")
![alt text](figures/network_training/MSNet5_big/metrics_fourier.png "Residuals")

![alt text](figures/network_training/MSNet5_big/losses_random.png "Losses")
![alt text](figures/network_training/MSNet5_big/metrics_random.png "Residuals")


## Weights of the different losses
Everything is run on random_8 and MSNet5_big for now:
| Loss weights (Inside, Dirichlet, Laplacian) | Remarks |
| ------------ | --------|
| (1.0, 1.0, 0.2) | Set used since the beginning |
| (0.0, 0.0, 1.0) | Opposite case (full_lapl) |
| (0.5, 0.5, 1.0) | More laplacian (weight_1) |
| (1.0, 1.0, 1.0) | More laplacian (weight_2) |

# UNet