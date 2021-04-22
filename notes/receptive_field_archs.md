# Summary of networks with fixed receptive field

Different Unet and MSNet networks are studied, fixing the Receptive field size
of the output of the coarsest scale. The architectures are found on:

'/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/archs'

## 100x100 Receptive field

### Unet (file: unets_RF_100.yml)

#### Unet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| UNet3        |  100 081           | 101x101     | [n = 5 , n/2 = 5, n/4 =  5]|


#### Unet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| UNet4        |   104 569         | 102x102     | [n = 3, n/2 = 3, n/4 = 2, n/8 = 2]   |


#### Unet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| UNet5        |     103 127       | 101x101     | [n = 1, n/2 = 2, n/4 = 1, n/8 = 1, n/16 = 1]   |


### Msnet (file: msnets_RF_100.yml)

#### Msnet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| MSNet3       |    101 367        | 101x101     | [n = 8 , n/2 = 7, n/4 =  7]|


#### Msnet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| MSNet4       |    102 114        | 105x105     | [n = 4, n/2 = 4, n/4 = 4, n/8 = 3]   |


#### Msnet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| MSNet5       |     101 677       | 125x125     | [n = 2, n/2 = 2, n/4 = 2, n/8 = 2, n/16 = 2]   |


## 200x200 Receptive field

### Unet (file: unets_RF_200.yml)

#### Unet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| UNet3        |     102 903       | 201x201     | [n = 10, n/2 = 10, n/4 =  10]|


#### Unet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| UNet4        |       99 373      | 206x206     | [n = 5, n/2 = 5, n/4 = 4, n/8 = 5]   |


#### Unet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| UNet5        |    103 185        | 201x201     | [n = 2, n/2 = 2, n/4 = 3, n/8 = 2, n/16 = 2]   |


### Msnet (file: msnets_RF_200.yml)

#### Msnet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| MSNet3       |       99 231      | 201x201     | [n = 14 , n/2 = 15, n/4 =  14]|


#### Msnet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| MSNet4       |       100 158     | 203x203     | [n = 7, n/2 = 7, n/4 = 6 n/8 = 7]   |


#### Msnet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| MSNet5       |     100 605       | 201x201     | [n = 4, n/2 = 4, n/4 = 4, n/8 = 3, n/16 = 3]   |



## 400x400 Receptive field

### Unet (file: unets_RF_400.yml)

#### Unet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| UNet3        |    101 139        | 401x401     | [n = 20 , n/2 = 20, n/4 =  20]|


#### Unet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| UNet4        |        100 493    | 402x402     | [n = 10, n/2 = 9, n/4 = 9 n/8 = 9]    |


#### Unet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| UNet5        |    104 023        | 401x401     | [n = 4, n/2 = 4, n/4 = 4, n/8 = 4, n/16 = 5]   |



### MSnet (file: msnets_RF_400.yml)

#### Msnet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| MSNet3       |       102 473     | 401x401     | [n = 28 , n/2 = 28, n/4 =  29]|


#### Msnet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| MSNet4       |     102 618      | 401x401     | [n = 14, n/2 = 13, n/4 = 14, n/8 = 13]    |


#### Msnet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| MSNet5       |       104 983     | 401x401     | [n = 6, n/2 = 7, n/4 = 7, n/8 = 7, n/16 = 6]   |
