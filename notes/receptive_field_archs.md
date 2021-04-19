# Summary of networks with fixed receptive field

Different Unet and MSNet networks are studied, fixing the Receptive field size
of the output of the coarsest scale. The architectures are found on:

'/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/archs'

## 100x100 Receptive field

### Unet (file: unets_RF_100.yml)

#### Unet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| UNet3        | 101 645           | 100x100     | [n = 4 , n/2 = 4, n/4 =  4]|


#### Unet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| UNet4        | 106 791           | 104x104     | [n = 1, n/2 = 1, n/4 = 2, n/8 = 2]   |


#### Unet 5

Not possible!


## 200x200 Receptive field

### Unet (file: unets_RF_200.yml)

#### Unet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| UNet3        | 103 997           | 196x196     | [n = 8 , n/2 = 8, n/4 =  8]|


#### Unet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| UNet4        | 105 509           | 200x200     | [n = 3, n/2 = 3, n/4 = 3 n/8 = 3]   |


#### Unet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| UNet5        | 106 411           | 208x208     | [n = 1, n/2 = 1, n/4 = 2, n/8 = 1, n/16 = 2]   |


## 400x400 Receptive field

### Unet (file: unets_RF_400.yml)

#### Unet 3

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $       |
| ------------ | ----------------- | ----------- | -------------------------- |
| UNet3        | 107 461           | 404x404     | [n = 16 , n/2 = 16, n/4 =  18]|


#### Unet 4

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                 |
| ------------ | ----------------- | ----------- | ------------------------------------ |
| UNet4        | 109 993           | 408x408     | [n = 6, n/2 = 6, n/4 = 6 n/8 = 7]    |


#### Unet 5

| Architecture | $n_\text{params}$ | $RF$        | $ Layers per scale $                           |
| ------------ | ----------------- | ----------- | ---------------------------------------------- |
| UNet5        | 109 703           | 400x400     | [n = 3, n/2 = 2, n/4 = 2, n/8 = 2, n/16 = 3]   |
