# Summary of the networks studied

## `51x51` training datasets

## `101x101` training datasets

### `config_1`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_L$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 0.2         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK        |
| `random_16`        | OK        |
| `random_4`         | OK        |

### `config_2`

Network to compare the influence of the physical loss compared to a classical points loss

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_I$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 1.0         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | OK |
| `random_16`        | OK |
| `random_4`         | OK |

### `config_3`

UNet5 with a deconvolution when going up, `up_type='deconvolution'`

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_I$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 1.0         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |
| `random_16`        | *Running* |
| `random_4`         | *Running* |

### `config_4`

UNet4 run with new architecture

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_I$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet4        | 420 000           | 1.0         | 1.0         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

### `config_5`

UNet3 run with new architecture

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_I$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet3        | 420 000           | 1.0         | 1.0         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |

### `config_6`

UNet5 with a wider receptive field (only)

| Architecture | $n_\text{params}$ | $\lambda_D$ | $\lambda_I$ |
| ------------ | ----------------- | ----------- | ----------- |
| UNet5        | 420 000           | 1.0         | 1.0         |

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |
| `random_16`        | *Running* |
| `random_4`         | *Running* |

## `201x201` training datasets