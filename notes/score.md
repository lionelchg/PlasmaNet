# Score of the neural network

The residual of the network is computed over a single dataset (more diverse than the training dataset) to produce a score:

$$
    S(x, N) = \frac{1}{n} \sum_i |x_\text{out}^i - x_\text{target}^i| \quad \text{for} \quad x \in \{\phi, \vb{E}, \nabla^2 \phi\}
$$

How do we compose the dataset for this score?

1. One third of `random`
2. One third of `fourier`
3. One third of `hills`