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

### Hyperparameters tunning

No scaling factor has been applied this time. The RHS is 1e8 approximtely and the potential is around unity.

| Network | Dataset    |
| ------- | ---------- |
| UNet5   | `random_8` |

#### Laplacian loss weight

With `random_8`:

| $\lambda_L$ | Name of run                | Observations                                                                                                                         |
| ----------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 2e+7        | -                          | Value previously used for 2D dirichlet square problem, it seems a bit high as the loss has exploded to 1e+20 on the first run        |
| 2+1         | -                          | Value too low, at epoch 20 the plots show that the network hasn't learned a thing                                                    |
| 2e+5        | `train_cyl/lapl_weight_1/` | Seems to be OK, learning seems slower due to the need for the network to understand that nodes around the axis have amplified values |
| 2e+6        | `train_cyl/lapl_weight_2/` | 100 epochs - print/save every 10 |