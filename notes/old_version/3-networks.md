# Directory: `outputs/3-networks/`

# Best network so far

## UNet5
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

Best loss for now:
$$
    \mathcal{L} = \lambda_D \mathcal{L}_D + \lambda_L \mathcal{L}_L
$$

with $\lambda_D = 1.0$ and $\lambda_L = 0.2$

# Does the previous study adapt to higher number of points?
Try to see if the intuition is right and that in this case `UNet6` should be better than `UNet5`.

## UNet6
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

## Equivalence of the `random` datasets

Since the grid is four times bigger in $201 \times 201$:

| Dataset $101 \times 101$ | Dataset $201 \times 201$ |
| ------------------------ | ------------------------ |
| `random_4`               | `random_8`               |
| `random_8`               | `random_16`              |
| `random_16`              | `random_32`              |

# Configurations test
## `config_1`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 0.2         |

### Training
| Dataset   | Results |
| --------- | ------- |
| random_8  |         |
| random_16 | OK      |
| random_32 | OK      |

### Running cases
Try on `gaussian`, `gaussian_offcenter`, `two_gaussians` on three periods.

## `config_2` and `config_4`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet6        | 420 000           | 1.0         | 0.2         |

Two different runs are made to assess the sensibility to initial coefficients.
`config_4` *Running*

### Training
| Dataset     | `config_2`                                                                                                                                                       | `config_4` |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `random_4`  |                                                                                                                                                                  |   |
| `random_8`  |                                                                                                                                                                  |   |
| `random_16` | Better than `config_1` (expected), confirms the interpretation, still the case after 500 epochs, slightly better in terms of losses and metrics than `random_32` |   |
| `random_32` | Better than `config_1` (expected), confirms the interpretation, still the case after 500 epochs                                                                  |   |

The losses for the LaplacianLoss are better when the resolution of the random grid is finer (from 32 to 4), the opposite tendency is found for the DirichletBoundaryLoss. The regularity of the solution is better ensured with a lot of points. Moreover the training seems to be more stable and less prone to random spikes.

Concerning the comparison between `config_2` and `config_4`, on the losses and metrics there is no clear better training. It should be interesting to see how errors in the training impact the simulation results in the end.

### Running cases

Try on `gaussian`, `gaussian_offcenter`, `two_gaussians` on three periods.

| Dataset   | `config_2`                                                                            | `config_4` |
| --------- | ------------------------------------------------------------------------------------- | ---------- |
| random_8  | *Running*                                                                             |
| random_16 | OK                                                                                    |
| random_32 | Seems to be the best for the stabilization of the networks, maybe it's just by chance |

## `config_3`
| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5-gen    | 420 000           | 1.0         | 0.2         |

| Dataset   | Results                                                                                     |
| --------- | ------------------------------------------------------------------------------------------- |
| random_8  |                                                                                             |
| random_16 | Bad results, the losses don't decrease from epoch 100 and we get stuck into a local minimum |
| random_32 | Same as random_16                                                                           |

This configuration starts to work on the $n_{1/2}$ scale and has the same architecture as the classical UNet5 network. This should accelerate the run compared to `config_2` without impacting the results if understanding of the network is good. Results are bad suggesting that although not a lot of weights are needed the finest scale is still necessary in the networks.