# Article notes

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