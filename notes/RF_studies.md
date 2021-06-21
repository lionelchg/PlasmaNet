# Summary of all the *_RF_study trainings

All the previous `*_RF_study/` directories are now found in:
`/scratch/cfd/PlasmaDL/networks/train/old_studies/`

The important trainings of these studies have thus been regrouped in the following directory:

`/scratch/cfd/PlasmaDL/networks/train/RF_study/`


All the trainings for now are made at a resolution of 101x101. In this folder, 4 main training families are found:

- Unet networks (`Unet`)
- Msnet networks (`Msnet`)
- Longer trainings (`Long_trainings`)
- Trainings using the inside loss (`Inside_loss`)

The logical structure to follow inside these networks is:

`Network/scales/kernel/RF/models/dataset` (i.e. `/scratch/cfd/PlasmaDL/networks/train/RF_study/Unet/5_scales/k_3/RF_200/models/random_8`)

## Unet networks

### 3 scales

#### kernel size 3

RF_fields: 50, 75, 100, 150, 200

#### kernel size 5

RF_fields: 100, 150, 200

### 4 scales

#### kernel size 3

RF_fields: 50, 75, 100, 150, 200

#### kernel size 5

RF_fields: 100, 150, 200

### 5 scales

#### kernel size 3

RF_fields: 100, 150, 200

#### kernel size 5
RF_fields: 200

## Msnet networks

#### kernel size 3

RF_fields: 50, 75, 100, 150, 200

#### kernel size 5
RF_fields: 75, 100, 150, 200

### 4 scales

#### kernel size 3

RF_fields: 50, 75, 100, 150, 200

#### kernel size 5

RF_fields: 150, 200

### 5 scales

#### kernel size 3

RF_fields: 150, 200

## Longer trainings

Unet = 3 networks with kernel size = 3 for RF = 50, 75, 100, 150, 200.

## Inside Loss

Unet = 3 networks with kernel size = 3 for RF = 50, 75, 100, 150, 200.