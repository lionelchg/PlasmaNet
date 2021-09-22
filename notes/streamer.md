# Notes on streamer simulation using NNs

## Linear system runs in `CfdSolver/scalar/`

The domain is of size 4 x 1 mm$^2$.

`dh_streamer_1`: 401 x 101, no photo
`dh_streamer_2`: 401 x 101, photo
`dh_streamer_3`: 801 x 201, no photo
`dh_streamer_4`: 801 x 201, photo

## Training UNets for cylindrical runs with double headed streamer

### Datasets

Creation of training datasets with `rhs_random_cyl.py`. The typical value of `n0` has been raised to $10^{16}$ m$^3$ for typical value of charge difference in the streamer cases. The random values from RHS are amplified around the axis so that there is a shift of values and the learning procedure seems to be harder with this kind of dataset. Think about a kind of dataset that will make the network understand this behavior of the Poisson equation in cylindrical coordinates that is different from cartesian coordinates.

### Modifications

Compared to the squared Dirichlet test case:

- Dirichlet Boundary loss only on: top left right
- Axial Loss added (similar to Inside Loss but only on the axis)
- Custom padding added (even if not great for now ...)

### Hyperparameters tunning

No scaling factor has been applied this time. The RHS is 1e8 approximtely and the potential is around unity.

| Network | Dataset    |
| ------- | ---------- |
| UNet5   | `random_8` |

#### Laplacian loss weight

##### 101x101 dataset

The following results correspond to the 101x101 streamer dataset, which is first used as a simpler version to tune the network. For now, the parameters which seem to yield the best results are:

| Laplacian | Dirichlet | Axial |
| --------- | --------- | ----- |
| 2e9       | 0         | 1e-4  |

After several tests, the following conclusions have been achieved:

- Higher Laplacian losses tend to work better (compared to 2e8 for example)
- Activating the 3 losses is not too benefitial, as the DirichletBoundary loss tends to over constrain the network (That's why the Dirichlet loss is equal to 0)
- Higher Axial losses (~1 are not too benefitial, as the network convergence is slower)
- Using only the Laplacian loss results in an instable network

These conclusions are based on the runs found in:

`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/train_cyl/debug/figures/UNet5_rf200`

Particularly, we can focus on the following:

| Name                         | Laplacian | Dirichlet | Axial |
| ---------------------------- | --------- | --------- | ----- |
| `basic_train`                | 2e9       | 1e-4      | 1e-4  |
| `no_bc_train`                | 2e9       | 0         | 0     |
| `only_axial_bc_train`        | 2e9       | 0         | 1e-4  |
| `only_axial_bc_train_weight` | 2e9       | 0         | 1     |
| `only_dir_bc_train`          | 2e9       | 1e-4      | 0     |

For now, the **best network seems to be `only_axial_bc_train`.** But there is still some improvement margin.

As the values were rather high, a new test was carried out with a new scaling factor, `only_axial_bc_train_no_scalingfactor`, where the scaling factor is 1.0 (previously fixed to 1e6). First results show that this is not a good idea, as the residual is higher.

Note that for this particular case, **custom padding** does not yield better results.


##### 401x101 dataset

The following results correspond to the 401x101 streamer dataset, thus matching the targeted resolution.

After several tests, the following conclusions have been achieved:

- For the rectangular domain, the **custom padding** seems benefitial (as results improve considerably)
- The Dirichlet Boundary loss seems to over-constrain the problem, as while the laplacian loss decreases, it increases or remains constant ... This results should be further investigated.

These conclusions are based on the runs found on:

`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/train_cyl/rectangle_debug/figures/UNet5_rf200`

Particularly, we can focus on the following:

- Traditional padding (zeros):

| Name                  | Laplacian | Dirichlet | Axial |
| --------------------- | --------- | --------- | ----- |
| `basic_train`         | 2e9       | 1e-4      | 1e-4  |
| `no_bc_train`         | 2e9       | 0         | 0     |
| `only_axial_bc_train` | 2e9       | 0         | 1e-4  |
| `only_dir_bc_train`   | 2e9       | 1e-4      | 0     |

- Custom Padding:

| Name                      | Laplacian | Dirichlet | Axial |
| ------------------------- | --------- | --------- | ----- |
| `basic_train_pad`         | 2e9       | 1e-4      | 1e-4  |
| `no_bc_train_pad`         | 2e9       | 0         | 0     |
| `only_axial_bc_train_pad` | 2e9       | 0         | 1e-4  |
| `only_dir_bc_train_pad`   | 2e9       | 1e-4      | 0     |


For now, there is no clear best network, although the **custom** padding networks seem to considerably outperform the others.

Try Neumann loss

###### `UNet5-rf200`

| Name    | $\lambda_D$ | $\lambda_{AD}$ | $\lambda_{AN}$ | $\lambda_L$ | padding | Comments                                                                                                                                       |
| ------- | ----------- | -------------- | -------------- | ----------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_1` | 0.0         | 1e-4           | 0              | 2e9         | custom  | Loss OK                                                                                                                                        |
| `run_2` | 0.0         | 0              | 1.0            | 2e9         | custom  | Explodes (maybe no potential ref)                                                                                                              |
| `run_3` | 1e-4        | 0.0            | 1e-4           | 2e9         | custom  |
| `run_4` | 1e-4        | 1e-4           | 0.0            | 2e9         | custom  |
| `run_5` | 1e-2        | 0.0            | 1e-2           | 2e9         | custom  | Best run so far up to 100 epochs. After restart the losses and metrics oscillate and the network does not seem to be able to learn any better. |
| `run_6` | 1e-2        | 1e-2           | 0.0            | 2e9         | custom  |
| `run_7` | 1e-2        | 0.0            | 1e-2           | 2e9         | custom  | Same run as `run_5` but without restart straight to 300 epochs                                                                                 |
| `run_8` | 1e-2        | 0.0            | 1e-2           | 2e9         | custom  | With UNet5-100k and rectangular kernel sizes                                                                                 |
| `run_9` | 1e-2        | 0.0            | 1e-2           | 2e9         | custom  | With UNet5-300k and rectangular kernel epochs                                                                                 |

###### `UNet5-ksrect`

| Name    | $\lambda_D$ | $\lambda_{AD}$ | $\lambda_{AN}$ | $\lambda_L$ | padding | Comments                                                                                                                                       |
| ------- | ----------- | -------------- | -------------- | ----------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_1` | 1e-2        | 0.0            | 1e-2           | 2e9         | custom  | With UNet5-100k and rectangular kernel sizes                                                                                 |
| `run_2` | 1e-2        | 0.0            | 1e-2           | 2e9         | custom  | With UNet5-300k and rectangular kernel epochs                                                                                 |