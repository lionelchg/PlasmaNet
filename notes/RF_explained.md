# Notes on Receptive field

For this paper a Receptive Field (RF) definition is proposed, which matches the definition found on the literature for the receptive field. Technically, a networks receptive field is described as:

"The receptive field in Convolutional Neural Networks (CNN) is the region of the  input space that affects a particular unit of the network" (https://blog.christianperone.com/2017/11/the-effective-receptive-field-on-cnns/)

Particularly, in this study is considered as the region of the input space that affects a chosen pixel of the output space. For feed-forward CNNs, where no upsampling/downsampling operations are performed, calculating the RF is trivial. However, when dealing with more complex networks that contain various upsampling and downsampling operations, concatenting fields with different spatial resolutions, computing the RF can be slightly more tricky. 

## Numerically computing the RF

 Based on the jupyter notebook (https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb), the RF can be easily computed numerically. If the weights of a network are initialized to 1 (and the biases to 0) and a field containing 1 (noted $I_{nn}$) is forwarded, a given output will be obtained (denoted $O_{nn}$):

 $$
 O_{nn} = f(I_{nn})
 $$

where $f()$ represents the NN. A dummy loss function can be introduced, which states that the loss function is 0 everywhere except in a point in the domain:

$$
\mathcal{L(i,j)} = \begin{cases}
               0, ~\forall (i,j)~i \neq i_{0}, j \neq j_{0} \\
               1,~i=i_0, j=j_0 
            \end{cases}
$$

Thus, if this loss is backpropagated through the network up until $I_{nn}$, the gradients at the entry level should be zero everywhere except in the zone of influence of the chosen point $(i_0, j_0)$. Thus if the non-zero points of the gradient field are counted, they should be equal to the size of the receptive field. 

## Inverse RF field definition

Even if slightly off-topic, the capability of the network to propagate information can be studied as well. Even if this definition does not match the RF, this information can be interesting for understanding for example how far the information coming grom the BC can be expanded. To compute this value, a neural network is still initialized with 1s (weights) and 0s (biases). If an ampty field is inputed which onmy has one pixel = 1.0, the network output will show a box which represents the zone of propagation of the network. 

Please note that even if the input is =1 and all the weights are equal to 1, the first FM will contain a field o 0s and 9s (for a 3x3) kernel. Thus, a geometric expansion is found, which means that the network output won't be = 1s and 0s but rather to some very high numbers. To avoid numerical issues double precision is used, and then the non zero values are converted to 1s!