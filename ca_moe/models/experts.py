"""
Expert models for Cost-Aware Mixture of Experts.

Plain nn.Module subclasses — no base class required.
Computational cost is handled externally by cost functions (ca_moe.factories.cost_functions).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressor(nn.Module):
    
    def __init__(self, input_size: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        return self.linear(x)


class NN1(nn.Module):    
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class NN2(nn.Module): 
    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, int],
        num_classes: int
    ):
        super().__init__()
        self.input_size = input_size
        hidden1, hidden2 = hidden_sizes
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    """Classic CNN architecture adapted for MNIST-like datasets."""
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6,
            kernel_size=5,
            stride=1,
            padding=2  # 'same' padding to maintain spatial dimensions
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=0  # 'valid' padding
        )
        
        # pooling layers
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # fully connected layers
        # for 28x28 input: after conv1+pool -> 14x14, after conv2+pool -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Activation
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # block 1
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        
        # block 2
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.avgpool(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fully connected layers
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        
        return x
