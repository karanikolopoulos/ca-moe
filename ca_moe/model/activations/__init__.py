import torch.nn as nn

activations = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax(dim=-1),
    "gelu": nn.GELU(),
    "linear": nn.Identity(),
}
