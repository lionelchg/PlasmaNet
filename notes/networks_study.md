# Summary of the networks studied

## `101x101` training datasets

All in `NNet/train/` on kraken.

### `up_type=upsample`

#### `config_1`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |

#### `config_2`

Network to compare the influence of the physical loss compared to a classical points loss

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |

#### `config_3`

UNet5 with a deconvolution when going up, `up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |


### `uptype=deconvolution`

#### `config_4`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |

#### `config_5`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |

#### `config_6`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

### Wider UNets for same number of parameters

This is wider so that the receptive field contains all the image

#### `config_7`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |

#### `config_8`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |

#### `config_9`

`up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 100 000           | 1.0         | 2.0e+3         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |