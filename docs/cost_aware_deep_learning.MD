##### Objective
Our goal is to incorporate the computational cost of a deep learning network as an extra minimization objective in the training procedure. We assume that the input space can be divided into easy and hard examples; hence the underlying complexity these networks possess, is not necessary for all input instances. Our conjecture is that we can achieve comparable performance by using a simpler model and therefore less resources. A gating network, in the form of a router, will be able to decide which input should be allocated to which model.


##### Proposed Solution: Mixture of Experts
Mixture of Experts (MoE) is an ensemble learning method, proposed by [Jacob et al., 1991](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf), that seeks to address regression and classification tasks in terms of subtasks using expert models. The idea behind MoE is mainly based on the divide and conquer principle, where you can build specialized models that will deal more effectively with certain parts of the space. In its traditional setting, MoE consists of several networks, called experts, of the same architecture. A gating network, usually a linear model or a simple neural network, makes soft partitions of the input space and outputs a propability for each expert. The loss function of MoE is a weighted combination of the losses incurred by the expert models, weighted by the probability of the gating model to assign an instance to each expert. The training aims to minimize this loss by learning the parameters of the expert models, and the parameters of the gating model. In this way we simultaneously learn how to partition the input space, and how to train the best model for each partition. At inference time, as opposed to other ensemble models, we aggregate the output of one, or a few experts as the final prediction.

Assuming we have $K$ expert models, let $x$ denote an input instance and $z_{i}(x)$ the outputs of the final layer of the gating network. These values are transformed into probabilities using a soft-max operator, so we obtain:
$$ p_{i} = \frac {e^{z_{i}}}{\sum \limits_{i}^{K} {e^{z_{i}}}} $$
Let now $L_{i}(x)$ denote the loss incurred by expert model $i$ on input instance $x$. The loss of the MoE is defined as:
$$ L_{x} = \sum \limits_{i=1}^{K} p_{i}L_{i(x)} $$

##### Cost-Aware Mixture of Experts
In this variation, different from the traditional setting, the specialized models have different architectures, and correspond to models of different complexity. The loss function for each expert also incorporates a computation cost term, that depends on the complexity of the model. The goal is to learn the parameters again, taking into account this computational cost. Since we are trying to minimize this cost, at inference time, we use the model stochastically. This means we select the expert model with probability proportional to that produced by the gating network, and run only the selected model.

Let $C_{i}$ denote the cost of expert ${i}$. The loss of the CA-MoE is defined as:
$$ L_{x} = \sum \limits_{i=1}^{K} p_{i}(L_{i(x)} + C_{i}) $$
The derivative with respect to the output of the expert models does not depend on the cost. The derivative with respect to the output values $z_i(x)$ is worked out as follows:
$$ \quad \frac {\partial {L(x)}}{\partial {z_{i}}} = p_{i} (L_{i}(x) + C_{i} - L(x)) $$
We can take the derivatives of the functions for $z_{i}(x)$, $L_{i}(x)$ with respect to the model weights, and we can then perform gradient descent. The gradients are essentially the same as with the traditional mixture of experts model, except for the additional term $C_{i}$ which models the computation cost. We want to raise the probabilities $p_{i}$ for all experts that give less than the weighted combination of all the experts. If expert $i$ makes a lower error than the weighted combination of all experts then we will try to raise the probability of expert $i$. Respectively, if expert $i$ makes a higher error, then we will try to lower its probability. In reality, it is this additional term that has a significant role in the weights, and therefore probabilities outputted by the gating network.

The cost can be considered as a penalty term in the CA-MoE loss. Experts with a low error and high cost should be punished, while experts with a low error and low cost should be favored. An expert with a low error and high cost should only be favored, when a low cost expert is not adequate, in terms of its high error. In the former case, we can assume that this input sample comes from an easier regime of the input space, while in the latter it comes from a difficult, or more difficult regime. This way, we achieve a trade-off between accuracy and cost, while also partitioning the input space.

#### Multi-Task Learning
Adding the computational cost as an objective into the MoE loss function, can be viewed as a MTL (Multi-Task Learning) problem. Typically, a MTL problem consists of two or more tasks. In this approach, instead of assigning each task to a model, we try to learn multiple related tasks simultaneously by a shared model, while utilizing both task-specific similarities and differences. This is achieved, by training tasks in parallel while using a shared representation.

There are two tasks in our case, the accuracy and the cost. In a MTL scenario the loss of the CA-MoE can be rewritten as:
<!-- $$
\begin{equation*}
    \begin{split}
        &L_{x} = \sum \limits_{i=1}^{K} p_{i}(L_{i(x)} + C_{i}) \\
        \implies &L_{x} = \sum \limits_{i=1}^{K} (p_{i}L_{i(x)} + p_{i}C_{i}) \\
        \implies &L_{x} = \sum \limits_{i=1}^{K} p_{i}L_{i(x)} + \sum \limits_{i=1}^{K} p_{i}C_{i} \\
        \implies &L_{x} = L_{x}^{accuracy} + L_{x}^{cost}
    \end{split}
\end{equation*}
$$ -->
<p align="center">
  <img src="eq_1" alt="Loss function equation" width="500"/>
</p>
Thus, the CA-MoE loss can be decomposed into two losses. The first loss, is the traditional loss of the original MoE model, regarding the accuracy. In this case, the gater selects the best or most capable experts in order to partition the input space and maximize the accuracy. The second loss is the weighted sum of the costs of every expert. In this case, the gater selects the expert that has the lowest computational cost. It is evident, that the probabilities $p_{i}$ take place in both losses, therefore the common backbone they share are the weights of the gating network.

The most common problem in a MTL scenario is that the losses may have different magnitudes. If not weighted properly one of the losses can take control of the training and the results are not always as intented.