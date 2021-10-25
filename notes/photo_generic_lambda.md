# Notes about photo ionization with variable lambda

The following note describes the procedure to train a cylindrical Unet network with
a variable lambda for photoionization. The concrete equation that the network is trained to solve is:

$$
\begin{aligned}
\nabla^2 \phi - \lambda^2\phi = -\psi R
\end{aligned}
$$

It should be noted that in previous trainings (NeurIps workshop) $\lambda$ and $\psi$ were fixed to:

|       case        |   $\lambda$ |         $\psi$       |
| ----------------- | ----------- | -------------------- |
|  Poisson equation |       0     |           1          |
|        J1         |    1461.0   |   4.73 $\times 10^5$ |
|        J2         |    8815.5   |   3.99 $\times 10^7$ |

The rhs scalind factor $\psi$ is not that important for this study case, as the main parameter
to be account is $\lambda$. Small values of $\lambda$ will result in a classical homogeneous
Poisson equation where the R field is diffused, whereas high $\lambda$ values will tend to
just scale the R field (resulting in topologically similar fields for R and $\phi$).


The config file on which the networks are trained is found at:

`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/train_photo_cyl_generic.yml`

The trained network for this study are found at:

`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/train_cyl/study_photo/generic_photo`

These networks were trained on a random 24 cylindric dataset found at:
`/scratch/cfd/cheng/DL/datasets/photo/train_cyl/401x101/random_24`
And they were trained with 100 k parameter Unet 5 found at 'unets_rect.yml'

## Simplest approach

In order to train the network with a variable $\lambda$ the simplest possible approach is taken,
which consists on inputting the $\lambda$ value as an extra input channel. Thus, at the trainer
(`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/PlasmaNet/nnet/trainer/trainer.py`) if the generic
photo loss is chosen, a second channel is added to data, including a $\lambda$ value. This value is
randomly chosen from an uniform distribution U(0,1), and is changed for every minibatch (see lines 110 and 247).
Thus, the new data tensor's shape is: (bsz, 2 , NNy, NNx). Note that when entering the network $\lambda$ varies
from 0-1 for coherence. However, the real value of lambda needs to be multiplied by 10^4 (phot scale factor
`lambda_scale` introduced in the config file and denormalized in the PhotoLoss).

### Data handling and normalization

The training procedure is quite susceptible to normalization, to a complete description will be now added.
The process is quite empirical, so it could be further discussed and changed.

1. *Entering the network:* The network takes the RHS field and $\lambda$ as inputs, and both values have
a unitary order of magnitude ($rhs_{max}$ $\simeq$ $\pm$ 2 and $\lambda_{max}$ = 1)
2. *Loss calculation:* When entering the loss, $\psi$ is applied so that values are more coherent.
For now:

$$
\begin{aligned}
\psi =  \alpha  (\lambda^2 + \frac{1}{dx \times dy})/ normalize
\end{aligned}
$$

where $normalize$ is set to $10^4$ in the yaml file. This value enables to get output values that are easier for the network to train.
If $\alpha=10$, $(\psi R)_{max} \simeq 2 \times 10^7$. To match this rhs, typical values of the network are: $ output_{max} \simeq \pm 5000 $
and $ output_{normalized-max} \simeq \pm 0.5 $ (as data_norm = $10^4$)

Note that lambda is de-normalized (multiplied by `lambda_scale` = $10^4$)

3. For plotting, as no previous $\phi$ is known (varies every minibatch), the correct $\phi$
   is calculated for every plotting call. At this point, lambda has already been denormalized,
   so just R needs to be factorized by $\alpha  (\lambda^2 + \frac{1}{dx \times dy})/ normalize$

   Note that the photo solver adds an scaling factor, thus the target needs to be multiplied by:
   $dx dy$ from the laplacian solver scaling factor. As the output is still $ output_{max} \simeq \pm 5000 $, for plotting we normalize it to  plot `output /data_norm`

### Results

The trained networks are found at

`/scratch/cfd/ajuria/Plasma/plasmanet_new/plasmanet/NNet/train_cyl/study_photo/generic_photo/figures/UNet5`

The performed runs correspond to:

|   name  |   alpha    | comments   |
| ------- | ---------- | ---------- |
| run_00  |     10     |  To check influence of $\alpha$ |
| run_01  |      1     |  Results  do not seem to change |
