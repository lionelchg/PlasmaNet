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


| Laplacian | Dirichlet  |   Axial  |
| --------- | ---------- |----------|
| 2e9       |     0      |    1e-4  |

After several tests, the following conclusions have been achieved:

- Higher Laplacian losses tend to work better (compared to 2e8 for example)
- Activating the 3 losses is not too benefitial, as the DirichletBoundary loss tends to over constrain the network (That's why the Dirichlet loss is equal to 0)
- Higher Axial losses (~1 are not too benefitial, as the network convergence is slower)
- Using only the Laplacian loss results in an instable network

These conclusions are based on the runs found on:

`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/train_cyl/debug/figures/UNet5_rf200`

Particularly, we can focus on the following:

|          Name                | Laplacian | Dirichlet  |   Axial  |
|------------------------------| --------- | ---------- |----------|
| `basic_train`                | 2e9       |   1e-4     |    1e-4  |
| `no_bc_train`                | 2e9       |     0      |    0     |
| `only_axial_bc_train`        | 2e9       |     0      |    1e-4  |
| `only_axial_bc_train_weight` | 2e9       |     0      |    1     |
| `only_dir_bc_train`          | 2e9       |     1e-4   |    0     |

For now, the **best network seems to be `only_axial_bc_train`.** But there is still some improvement margin.

As the values were rather high, a new test was carried out with a new scaling factor, `only_axial_bc_train_no_scalingfactor`, where the scaling factor is 1.0 (previously fixed to 1e6). First results show that this is not a good idea, as the residual is higher.