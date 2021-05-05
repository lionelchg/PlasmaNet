# 4_RF_study notes

Notes concerning the directory of networks contained in `4_RF_study/`

The goal is to add networks with receptive at 50 in 101 nodes, so half the size of the domain.
The intuition is that there would be a gap in performance compared to RF100 networks bigger than the differences between RF100 and RF200.

Kernel size of 5 is also tested for RF100 networks. For a given receptive field, the width of the network is smaller, so vanishing/exploding gradient problems are less likely to arise and the network should also be able to learn somewhat faster.

The networks are first trained on 101x101 domains.

## `unets_ks3_rf50.yml`

UNet5 is not possible with RF = 50.

### UNet3 - `config_1`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |

### UNet4 - `config_2`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |

## `unets_ks3_rf100.yml`

UNet5 is not possible with RF = 50.

### UNet3 - `config_3`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |

### UNet4 - `config_4`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |

### UNet5 - `config_5`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |

## `unets_ks5_rf100.yml`

UNet5 is not possible with RF = 50.

### UNet3 - `config_6`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |

### UNet4 - `config_7`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running*       |