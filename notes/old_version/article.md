# Networks

From previous studies, the runs will focus on the doubled runs of network architectures:

| NN name                    | Architecture | Training resolution | Training length |
| -------------------------- | ------------ | ------------------- | --------------- |
| `2-networks/config_[0, 1]` | UNet5        | 101                 | 0.01            |
| `2-networks/config_6` | UNet3        | 101                 | 0.01            |
| `2-networks/config_7` | UNet4        | 101                 | 0.01            |
| `2-networks/config_8` | UNet6        | 101                 | 0.01            |
| `3-networks/config_[2, 4]` | UNet6        | 201                 | 0.01            |

## `2-networks/config_1`

Six profiles are considered for each run: three gaussians and three random profiles with a gaussian mask.

### Influence of the dataset

### Influence of resolution

For each run, a gaussian, offcenter gaussian and two_gaussians are applied. The domain length is fixed at 1 cm.

| Training dataset / Resolution | 61                 | 81                 | 101                | 121                | 141                | 201                |
| ----------------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| `random_2`                    |                    |                    |                    |                    |                    |                    |
| `random_4`                    |                    |                    |                    |                    |                    |                    |
| `random_8`                    | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| `random_16`                   |                    |                    |                    |                    |                    |                    |

### Influence of length of the domain

For each run, a gaussian, offcenter gaussian and two_gaussians are applied. The domain length is fixed at 1 cm.

| Training dataset / Length | 0.01 | 0.001 | 0.1 |
| ------------------------- | ---- | ----- | --- |
| `random_2`                |      |       |     |
| `random_4`                |      |       |     |
| `random_8`                |      |       |     |
| `random_16`               |      |       |     |

## Potential and electric field study on the effect of the `random` datasets network, for different lengths, different resolutions, different profiles

Parametric study just in terms of correlation of rhs and potential/electric field 

Show validation cases, emphasize the importance of the scales and going from `UNet4` to `UNet5` and the link with the resolution studies and monitor performance

Prepare sets of 2000 snapshots of:

1. Gaussians (random for amplitude, number of gaussians)
2. Sin modes
3. Random dataset that the network never saw

For this pass of 2000 snapshots compare the time it takes linear system solvers with CPUs and the time taken by one GPU with one CPU. Try to evaluate an equivalence between GPU hour and CPU hours
For performance of GPU vs CPU ask Gab what would be the best choice for conversion of GPUs in terms of CPUs, try to plot number of snapshots vs time/nGPU time/nCPU

## Peformance runs to prepare with either Periodic/Neumann BC

Try the splitted domain with either Neumann or Periodic BC to see which one fits what we need.

## For article

1. Maybe change the layout with reorganization of the article to have less space concerning plasma oscillation.

## Focus points of the article

The method needs to verify three points:

1. The solution should be physical ($\phi$ and $\mathbf{E}$ should be OK with residuals)
2. The solution should yield fast results (exploit the batch_size of the network) and provide considerable speedups compared to linear solvers
3. The solution should be easy to setup for someone else
