# 6_RF_study notes

Notes concerning the directory of networks contained in `6_RF_study/`

The goal is to add networks with receptive at 150 in 200 nodes, so more than the size of the domain.
The goal of this study is to complement the previous 4_RF_study and we should see a stagnation of the results, as the RF should be already saturated.

Kernel size of 5 is also tested. For a given receptive field, the width of the network is smaller, so vanishing/exploding gradient problems are less likely to arise and the network should also be able to learn somewhat faster.

The networks are first trained on 101x101 domains.

## `unets_ks3_rf200.yml`


### UNet3 - `config_1`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |    OK     |
| `random_4`         |    OK     |

### UNet4 - `config_4`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |    OK     |
| `random_4`         |    OK     ||

### UNet5 - `config_7`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |     OK    |
| `random_4`         |     OK    |

## `unets_ks5_rf150.yml`

UNet5 is not possible with RF = 50.

### UNet3 - `config_2`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |    OK     |
| `random_4`         |    OK     |

### UNet4 - `config_5`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |    OK     |
| `random_4`         |    OK     ||

### UNet5 

UNet5 is not possible with RF = 150.|


## `unets_ks5_rf200.yml`

### UNet3 - `config_3`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |     OK    |
| `random_4`         | *Running* |

### UNet4 - `config_6`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         | *Running* |
| `random_4`         | *Running* |

### UNet5 - `config_8`

| Dataset            | Results   |
| ------------------ | --------- |
| `random_8`         |     OK    |
| `random_4`         | *Running* |
