# Directory: `outputs/2-networks/`

# Architectures 
The chosen networks for MSNet and UNet:

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

### MSUNet5
```python
def __init__(self, data_channels):
    super(MSUNet5, self).__init__()
    self.convN_1 = _ConvBlockIn(1, 32, 32)
    self.convN_2 = _ConvBlockDown(52, 32, 32)
    self.convN_3 = _ConvBlockDown(52, 32, 32)
    self.convN_4 = _ConvBlockDown(52, 48, 62)
    self.convN_5 = _ConvBlockDown(92, 60, 62)
    self.convN_6 = _ConvBlockUp(124, 64, 64)
    self.convN_7 = _ConvBlockUp(96, 64, 64)
    self.convN_8 = _ConvBlockUp(96, 32, 32)
    self.convN_9 = _ConvBlockUp(64, 32, 32)
    self.final = _ConvBlockOut(32, 1)
```

### MSNet5_rev
```python
def __init__(self, data_channels):
    super(MSNet5_rev, self).__init__()
    self.conv_16 = _ConvBlockLong(data_channels, 32, 64, 112, 1)
    self.conv_8 = _ConvBlockLong(data_channels + 1, 32, 64, 96, 1)
    self.conv_4 = _ConvBlockShort(data_channels + 1, 32, 64, 1)
    self.conv_2 = _ConvBlockShort(data_channels + 1, 32, 64, 1)
    self.conv_1 = _ConvBlockShort(data_channels + 1, 32, 64, 8)
    self.final = nn.Conv2d(8, 1, kernel_size=1)
```

Best loss for now:
$$
    \mathcal{L} = \lambda_D \mathcal{L}_D + \lambda_L \mathcal{L}_L
$$


# Configurations test
## `config_0` and `config_1`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 0.2         |

| Dataset                  | Results                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `random_8`               | Best with config_3                                                                                                          |
| `target_case`            | Best with config_3                                                                                                          |
| `random_16`              | The losses look a lot like random_8, still not the expected behavior, we would expect the network to be better in this case |
| `random_4`               | *Running* config_3                                                                                                          |
| `fourier_5_2`            | Good results for all the metrics                                                                                            |
| `fourier_5_4`            | Better results than fourier_5_2 (expected)                                                                                  |
| `sim_gaussian`           | OK when used in simulation it crashes however (unstable)                                                                    |
| `sim_gaussian_offcenter` | OK when used in simulation it crashes however (unstable)                                                                    |
| `sim_2gaussians`         | OK when used in simulation it crashes however (unstable)                                                                    |
| `sim_sines_2D2`          | OK when used in simulation it crashes however (unstable)                                                                    |
| `random_4_center`        | *Running*                                                                                                                   |
| `random_8_center`        | *Running*                                                                                                                   |
| `random_16_center`       | *Running*                                                                                                                   |

`config_0` and `config_1` have exactly the same parameters and loss weights, however they yield different results on target_case, so the local minimum that the two configurations found are not the same. `config_0` is more stable when comparing the target_case. All the other trainings are therefore doubled with `config_0` and `config_1`.

The best trainig datasets for all the simulations in terms of stability are the `random_8` and `random_16`. They are even more stable than the "identity cases" with an edge for `random_8`. Let us push the idea with `random_4` to see if it's even more stable?

## `config_2`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| MSNet5       | 420 000           | 1.0         | 0.2         |

| Dataset     | Results                                                                                   |
| ----------- | ----------------------------------------------------------------------------------------- |
| random_8    | Oscillates a lot more than the UNet-like architectures but in the end not too bad overall |
| target_case | Around epoch 100 the losses start to rise up unexpectedly...                              |

## `config_3`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| MSUNet5      | 456 000           | 1.0         | 0.2         |

| Dataset     | Results            |
| ----------- | ------------------ |
| random_8    | Best with config_3 |
| target_case | Best with config_3 |

## `config_4`
The number of parameters is reversed with respect to the scale, so the coarser grid will have a lot of parameters compared to the finer one, this is more reasonable when looking at the analytical solution, where the first modes are amplified.
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| MSNet5_rev   | 430 000           | 1.0         | 0.2         |

| Dataset     | Results                                                                                                                         |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------- |
| random_8    | Very bad results compared to the other networks, 2 orders of magnitude on the losses compared to `config_1` and `config_3`      |
| target_case | Results that are not so bad compared with random_8 but with still two orders of magnitude higher than `config_1` and `config_3` |

## `config_5`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ | $\lambda_E$ |
| ------------ | ----------------- | ----------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 0.0         | 1.0e+6      |

| Dataset     | Results                                                                                                            |
| ----------- | ------------------------------------------------------------------------------------------------------------------ |
| random_8    | Comparison with `config_1` show that the results of `config_5` are worse than `config_1` when looking at residuals |
| target_case | Same than `random_8`                                                                                               |

## `config_6`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 420 000           | 1.0         | 0.2         |

| Dataset  | Results   |
| -------- | --------- |
| random_8 | OK |
| random_8_center | *Running* |

## `config_7`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 420 000           | 1.0         | 0.2         |

| Dataset  | Results   |
| -------- | --------- |
| random_8 | OK |
| random_8_center | *Running* |

## `config_8`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet6        | 420 000           | 1.0         | 0.2         |

| Dataset  | Results   |
| -------- | --------- |
| random_8 | OK |
| random_8_center | *Running* |

# Plasma oscillation

## Results on gaussian

## Other profiles
More complicated profiles are possible such as sines:
$$
    n_e(x, y, 0) = n_0 \sin\left(\frac{n \pi x}{L_x}\right) \sin\left(\frac{m \pi y}{L_y}\right)
$$

It is also possible to take off center gaussians, add multiple gaussians etc...

Find the best datasets to train on to get good results on all kinds of gaussians first, then on gaussians and sines

