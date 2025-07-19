import logging

import torch
import torch.nn as nn

from dataclasses import dataclass

from ca_moe.model.constraints import get_constraint
from ca_moe.model.initializers import get_initializer
from ca_moe.model.regularizers import get_regularizer

logger = logging.getLogger(__name__)


@dataclass
class Dense:
    in_units: int
    out_units: int
    use_bias: bool = True
    activation: str = "relu"
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"
    kernel_regularizer: str | None = None
    bias_regularizer: str | None = None
    activity_regularizer: str | None = None
    kernel_constraint: str | None = None
    bias_constraint: str | None = None


@dataclass
class Conv2D:
    input_shape: tuple | list | None
    filters: int
    kernel_size: int
    strides: int = 1
    padding: str = "valid"
    data_format: str = None
    dilation_rate: int = 1
    groups: int = 1
    activation: str = None
    use_bias: bool = True
    kernel_initializer: str = "glorot_uniform"
    bias_initializer: str = "zeros"
    kernel_regularizer: str | None = None
    bias_regularizer: str | None = None
    activity_regularizer: str | None = None
    kernel_constraint: str | None = None
    bias_constraint: str | None = None


class InitMixin:
    def __init__(self, kernel_initializer: str, bias_initializer: str) -> None:
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        assert hasattr(self, "core"), "Parent class should define `core` nn.Module"

    def apply_initializers(self):
        with torch.no_grad():
            self.kernel_initializer(self.core.weight)
            if self.core.bias is not None:
                self.bias_initializer(self.core.bias)


class RegMixin:
    def __init__(
        self, kernel_regularizer: str, bias_regularizer: str, activity_regularizer: str
    ) -> None:
        self.kernel_regularizer = get_regularizer(kernel_regularizer)
        self.bias_regularizer = get_regularizer(bias_regularizer)
        self.activity_regularizer = get_regularizer(activity_regularizer)

        assert hasattr(self, "core"), "Parent class should define `core` nn.Module"

    def get_kernel_regularization_loss(self):
        return self.kernel_regularizer(self.core.weight)

    def get_bias_regularization_loss(self):
        return self.bias_regularizer(self.core.bias)

    def get_activity_regularization(self, output):
        return self.activity_regularizer(output)

    def get_total_regularization_loss(self, output: torch.Tensor) -> torch.Tensor:
        kernel_reg = self.get_kernel_regularization_loss()
        bias_reg = self.get_bias_regularization_loss()
        activity_reg = self.get_activity_regularization(output=output)

        logger.debug(f"Kernel regularization loss: {kernel_reg}")
        logger.debug(f"Bias regularization loss: {bias_reg}")
        logger.debug(f"Activity regularization loss: {activity_reg}")

        return kernel_reg + bias_reg + activity_reg


class ConstraintMixin:
    def __init__(
        self, kernel_constraint: str | None = None, bias_constraint: str | None = None
    ) -> None:
        self.kernel_constraint = get_constraint(kernel_constraint)
        self.bias_constraint = get_constraint(bias_constraint)

        assert hasattr(self, "core"), "Parent class should define `core` nn.Module"

    def apply_constraints(self):
        with torch.no_grad():
            self.core.weight.copy_(self.kernel_constraint(self.core.weight))
            self.core.bias.copy_(self.bias_constraint(self.core.bias))


class TorchDense(InitMixin, RegMixin, ConstraintMixin, nn.Module):
    """Torch equivalent of tf.keras.Dense"""

    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
        "gelu": nn.GELU(),
        "linear": nn.Identity(),
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "relu",        
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: str | None = None,
        bias_regularizer: str | None = None,
        activity_regularizer: str | None = None,
        kernel_constraint: str | None = None,
        bias_constraint: str | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.core = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        # mixins require that core attr exist
        InitMixin.__init__(
            self,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        RegMixin.__init__(
            self,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
        )
        ConstraintMixin.__init__(
            self,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )

        self.activation = self.activations[activation]
        self.apply_initializers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.core(x)
        output = self.activation(output)

        return output
