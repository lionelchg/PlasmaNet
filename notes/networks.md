# Summary of the networks studied

## `101x101` training datasets

All in `/scratch/cfd/PlasmaDL/networks/train/models/` on kraken.

### `up_type=upsample`

The architectures are taken from `NNet/archs/unets_small.yml`

#### `config_1`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |

#### `config_2`


| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |

#### `config_3`


| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |


### `uptype=deconvolution`

The architectures are taken from `NNet/archs/unets_small.yml`

#### `config_4`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |

#### `config_5`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |

#### `config_6`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

### Wider UNets for same number of parameters

This is wider so that the receptive field contains all the image for the bottom part of the architecture. The architectures are taken from `NNet/archs/unets_wide.yml`. The `upsample` method for the right part of the UNet is used.

#### `config_7`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |

#### `config_8`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |

#### `config_9`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |


### `uptype=deconvolution` 
Still using wide networks (same as 7-9, see `NNet/archs/unets_wide.yml`)

#### `config_10`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

#### `config_11`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

#### `config_12`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+7         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

