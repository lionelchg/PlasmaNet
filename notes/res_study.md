# Resolution sensibility of network

A study of the behavior of a network trained on a certain resolution and applied to other resolutions is studied here.

## Intuition

From a given $n_r$ resolution, I have the intuition that $\alpha n_r$ would be less precise than $n_r / \alpha$ (where $\alpha > 1$) because when there are more points the network needs to conjure up information from nowhere.

## Optimal network for each resolution

For a given resolution, there is a maximum number of branches $n_b$ that can be achieved. For $101 \times 101$ resolution it is $n_b = 5$ because the less resolved picture is $6 \times 6$. With $n_b = 6$ going to a $3 \times 3$ grid doesn't make much sense with a $k_s = 3$ kernel size.

So inference using a UNet5 on $51 \times 51$ should not yielg very good results. 

## Receptive field

We have seen from earlier studies that for a 101-trained network, going from 50 to 200 receptive field yields a continuous improvement of the network. At $\mrm{RF} = 200$ all points in the domain interact with each other (even from one boundary to the other). If the intuition is there then $\mrm{RF} = 300$ and $\mrm{RF} = 400$ networks should not give significant improvements from $\mrm{RF} = 200$. In that case why do we choose $\mrm{RF} = 2 \times n_x$ ?

If there is indeed a plateau when $\mrm{RF} > 2 n_x$ then if we want to go to higher resolution when training on lower resolution the target resolution should be the one giving the good receptive field, but even then since the information is not the same it is not sure...

$n_b$ max such that the smallest grid is bigger than the kernel size and $\mrm{RF} = 2 n_x$ is a sweet spot for a given resolution because it is accurate and the network with better performance. Increasing further the receptive field increases the depth of the network, yielding vanishing gradient problems and also performance degradation (to check?).

