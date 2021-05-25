# 7_RF_study notes

Notes concerning the directory of networks contained in `7_RF_study/`

The Unet 3 and 4 with kernel size = 5 seem to outperform the kernel 3 networks even if they have the same receptive field. Out intuition says that if trained for longer, the kernel 3 networks shopuld be just as good. Thus, the same kernel 3 networks are trained for 1000 epochs.

The networks are first trained on 101x101 domains.

## `unets_ks3_rf100.yml`

### UNet3 - `config_1`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_4`         | *Running* |
| `random_8`         |    OK     |
| `fourier_2_8_1`    |    OK     |
| `fourier_2_8_2`    |    OK     |

## `unets_ks3_rf150.yml`

### UNet3 - `config_2`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_4`         |    OK     |
| `random_8`         |    OK     |
| `fourier_2_8_1`    |    OK     |
| `fourier_2_8_2`    |    OK     |

## `unets_ks3_rf200.yml`

### UNet3 - `config_3`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_4`         | *Running* |
| `random_8`         |    OK     |
| `fourier_2_8_1`    |    OK     |
| `fourier_2_8_2`    |    OK     |

## `unets_ks3_rf50.yml`

### UNet3 - `config_4`

| Dataset            | Results   |
| ------------------ | --------- |
| `fourier_2_8_1`    |    OK     |
| `fourier_2_8_2`    |    OK     |


## `unets_ks3_rf75.yml`

### UNet3 - `config_5`

| Dataset            | Results   |
| ------------------ | --------- |
| `fourier_2_8_1`    |    OK     |
| `fourier_2_8_2`    |    OK     |