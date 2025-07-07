from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from dataclasses import dataclass

from ca_moe.model.regularizers import get_regularizer

@dataclass
class Dense:
    n_units: int
    activation: str="relu"
    use_bias: bool=True
    kernel_initializer: str="glorot_uniform"
    bias_initializer: str="zeros"
    kernel_regularizer: None
    bias_regularizer: None
    activity_regularizer: None
    kernel_constraint: None
    bias_constraint: None


# class InitMixin:
    
#     @dataclass
#     class Initializer:
#         glorot_uniform: callable = nn.init.xavier_uniform_
#         glorot_normal: callable = nn.init.xavier_normal_
#         zeros: callable = nn.init.zeros_
#         ones: callable = nn.init.ones_
#         normal: callable = partial(nn.init.normal_, mean=0.0, std=0.05)
    
#         def get(self, name: str) -> callable:
#             if hasattr(self, name):
#                 return getattr(self, name)
#             raise ValueError(f"Initializer {name} not found")
    
#     def __init__(self, kernel_initializer: str, bias_initializer: str) -> None:
#         super().__init__()
#         self.kernel_initializer = self.Initializer.get(kernel_initializer)
#         self.bias_initializer = self.Initializer.get(bias_initializer)


#     def apply_initializers(self):
#         assert hasattr(self, 'core'), "Parent class should define `core` nn.Module"
        
#         self.kernel_initializer(tensor=self.core.weight)
#         if self.core.bias is not None:
#             self.bias_initializer(tensor=self.core.bias)




# class RegMixin:    
#     def __init__(self, kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None):
#         super().__init__()
#         self.kernel_regularizer = kernel_regularizer
#         self.bias_regularizer = bias_regularizer
#         self.activity_regularizer = activity_regularizer
#         self.activity_reg_loss = 0.0

#     def compute_activity_regularization(self, output):
#         if self.activity_regularizer is not None:
#             self.activity_reg_loss = self.activity_regularizer(output)
#         else:
#             self.activity_reg_loss = 0.0

#     def get_kernel_regularization_loss(self):
#         if self.kernel_regularizer is not None and hasattr(self, 'core'):
#             return self.kernel_regularizer(self.core.weight)
#         return 0.0

#     def get_bias_regularization_loss(self):
#         if self.bias_regularizer is not None and hasattr(self, 'core') and self.core.bias is not None:
#             return self.bias_regularizer(self.core.bias)
#         return 0.0

#     def get_total_regularization_loss(self):
#         return (self.get_kernel_regularization_loss() + 
#                 self.get_bias_regularization_loss() + 
#                 self.activity_reg_loss)

# class TorchDense(nn.Module):
#     """Torch equivalent of tf.keras.Dense"""
    
#     @dataclass
#     class Activations():
#         relu: nn.ReLU
#         tanh: nn.Tanh
#         sigmoid: nn.Sigmoid
#         softmax: nn.Softmax
#         gelu: nn.GELU
#         linear: nn.Identity
    
#     def __init__(self, out_features: int, activation: str, bias: str, kernel_initializer: str, bias_initializer: str):
#         linear_module = nn.LazyLinear(out_features=out_features, bias=bias)
#         activation_module = getattr(self.Activations, activation)
        