# Target case
The size of the domain has been switched from 128 to 101 to be closer to the simulation
```
casename: 'runs/kraken/target_case/'

params:
  n_periods: 2
  geom: 'xy'
  n_back: 1.0e+16
  n_pert: 1.0e+11
  sigma: 1.0e-3
  dt: 1.0e-10
  nt_oscill: 5000

mesh:
  xmin: 0
  ymin: 0
  xmax: 1.0e-2
  ymax: 1.0e-2
  nnx: 101
  nny: 101

BC: 'full_out'

output:
  save: 'plasma_period'
  verbose: False
  period: 0.1
  files: 'fig'
  dl_save: 'yes'
```

All sizes of dataset are 10000 to begin.
# Random datasets
The value of the density associated to rhs is n = $10^{11}$ as in the target case
## random_4
Random dataset with initial random values taken on a 101 / 4 = 25 x 25 grid
## random_8
Random dataset with initial random values taken on a 101 / 8 = 12 x 12 grid
## random_16
Random dataset with initial random values taken on a 101 / 16 = 6 x 6 grid

# Random fourier datasets
The value of the density associated to rhs is n = $10^{11}$ as in the target case
## random_5
5 modes with no selection of frequencies

## random_5_2
5 modes with square decreasing frequency with n and m

## random_5_4
5 modes with power of 4 decreasing frequency with n and m

# Hills datasets

In this case the simulation dataset and the hills dataset are the same

# Simulation datasets

About precision the electric field is direclty under a divergence in the drift-diffusion model.
In the Euler equations the link is less straightforward, is there an impact for the precision?

## Plasma oscillation

| Name | Size | Content |
| - | - | - |
| `target_case` | 10000 | Original simulation case with centered gaussian on 2 periods |
| `sim_gaussian` | 5000 | Same as target_case but with half entries | 
| `sim_2gaussians` | 5000 | Two gaussian on x axis slightly off center |
| `sim_sines_2D2` | 5000 | Sines dataset with $n = m = 2$ |


Dataset with 10000 entries associated with the target case.

## Double headed streamer